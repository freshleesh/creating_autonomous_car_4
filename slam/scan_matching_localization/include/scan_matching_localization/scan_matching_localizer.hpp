// Base node for scan-to-map 2D LiDAR localization.
//
// I/O contract (identical to the particle_filter / cartographer modes so it
// drops straight into stack_master/launch/middle_level.launch.xml):
//   subscribes : /scan        (sensor_msgs/LaserScan, frame = laser)
//                /vesc/odom    (nav_msgs/Odometry,    translation prediction;
//                               only when use_odom:=true)
//                /vesc/sensors/imu/raw (sensor_msgs/Imu, heading prediction)
//                /map          (nav_msgs/OccupancyGrid, transient-local latch)
//                /initialpose  (geometry_msgs/PoseWithCovarianceStamped, RViz)
//   publishes  : <out>/pose/odom (nav_msgs/Odometry, pose in map frame) -> EKF
//                map -> base_link TF (only when publish_tf:=true, i.e. real HW)
//
// Pipeline per scan: predict the new pose, then refine it by registering the
// scan against the map with the algorithm supplied by the derived class (ICP
// or NDT). The translation prediction comes from wheel odometry, or -- when
// use_odom:=false -- from integrating the IMU linear acceleration between scans
// (dead-reckoning, with the velocity re-set from the LiDAR motion each scan),
// i.e. pure LiDAR + IMU. The heading prediction comes from the integrated IMU
// gyro when available.
#pragma once

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

namespace scan_matching_localization
{

struct Pose2D
{
  double x = 0.0;
  double y = 0.0;
  double theta = 0.0;
};

inline double wrapAngle(double a)
{
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

// result = base (+) delta, with delta expressed in base's body frame.
inline Pose2D compose(const Pose2D & b, const Pose2D & d)
{
  const double c = std::cos(b.theta), s = std::sin(b.theta);
  return {b.x + c * d.x - s * d.y,
          b.y + s * d.x + c * d.y,
          wrapAngle(b.theta + d.theta)};
}

// body-frame delta d such that to = from (+) d.
inline Pose2D relative(const Pose2D & from, const Pose2D & to)
{
  const double c = std::cos(from.theta), s = std::sin(from.theta);
  const double dxw = to.x - from.x, dyw = to.y - from.y;
  return {c * dxw + s * dyw,
          -s * dxw + c * dyw,
          wrapAngle(to.theta - from.theta)};
}

using Points = std::vector<Eigen::Vector2d>;

class ScanMatchingLocalizer : public rclcpp::Node
{
public:
  ScanMatchingLocalizer(const std::string & node_name, const std::string & default_out_topic)
  : rclcpp::Node(node_name)
  {
    scan_topic_ = declare_parameter<std::string>("scan_topic", "/scan");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/vesc/odom");
    imu_topic_ = declare_parameter<std::string>("imu_topic", "/vesc/sensors/imu/raw");
    map_topic_ = declare_parameter<std::string>("map_topic", "/map");
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    out_topic_ = declare_parameter<std::string>("output_odom_topic", default_out_topic);

    publish_tf_ = declare_parameter<bool>("publish_tf", true);
    seed_from_tf_ = declare_parameter<bool>("seed_pose_from_tf", false);

    // Motion model: heading from the IMU gyro when use_imu and IMU data are
    // available, else from the translation source's yaw.
    use_imu_ = declare_parameter<bool>("use_imu", true);
    imu_yaw_scale_ = declare_parameter<double>("imu_yaw_scale", 1.0);

    // Translation prediction: from /vesc/odom when true; when false the wheel
    // odometry is ignored entirely and translation is dead-reckoned by
    // integrating the IMU linear acceleration between scans (velocity re-set
    // from the LiDAR motion each scan), i.e. pure LiDAR + IMU.
    use_odom_ = declare_parameter<bool>("use_odom", true);

    // Debug helpers: re-publish the consumed map (so it is visualizable even
    // when no map server is around) and log per-scan registration timing.
    publish_map_ = declare_parameter<bool>("publish_map", true);
    map_out_topic_ = declare_parameter<std::string>("map_out_topic", "/map");
    debug_timing_ = declare_parameter<bool>("debug_timing", true);

    pose_.x = declare_parameter<double>("initial_x", 0.0);
    pose_.y = declare_parameter<double>("initial_y", 0.0);
    pose_.theta = declare_parameter<double>("initial_yaw", 0.0);

    min_range_ = declare_parameter<double>("min_range", 0.1);
    max_range_ = declare_parameter<double>("max_range", 15.0);
    max_iterations_ = declare_parameter<int>("max_iterations", 20);
    min_scan_points_ = declare_parameter<int>("min_scan_points", 30);
    occ_threshold_ = declare_parameter<int>("occupancy_threshold", 65);

    // ICP fitness gate: only trust (use) the LiDAR-corrected pose when the
    // registration is actually good. When the fit is below threshold the scan
    // match is unreliable (e.g. featureless corridor, kidnapped, bad map) — we
    // reject the ICP pose and coast on the motion prediction for that scan.
    //   accept  iff  inlier_ratio >= min_inlier_ratio  AND  mean_resid <= max_resid
    fitness_gate_enable_ = declare_parameter<bool>("fitness_gate_enable", true);
    fitness_min_inlier_ratio_ =
      declare_parameter<double>("fitness_min_inlier_ratio", 0.55);
    fitness_max_resid_ = declare_parameter<double>("fitness_max_resid", 0.12);

    // Dynamic-object filtering: remove scan points around a perception-tracked
    // opponent before registration, so its (moving) returns can't bias the
    // scan-to-(static)-map match. The opponent centre is consumed in the map
    // frame from `dynamic_obstacle_topic` and projected into the scan frame with
    // the predicted pose. use_dynamic_filter=false → original behaviour.
    use_dynamic_filter_ = declare_parameter<bool>("use_dynamic_filter", false);
    dynamic_filter_radius_ = declare_parameter<double>("dynamic_filter_radius", 0.60);
    dynamic_filter_timeout_ = declare_parameter<double>("dynamic_filter_timeout", 0.30);
    dynamic_obstacle_topic_ =
      declare_parameter<std::string>("dynamic_obstacle_topic", "/local_planning/opponent");

    // base_link -> laser extrinsic; replaced by a TF lookup once available.
    laser_x_ = declare_parameter<double>("laser_x", 0.27);
    laser_y_ = declare_parameter<double>("laser_y", 0.0);
    laser_yaw_ = declare_parameter<double>("laser_yaw", 0.0);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(out_topic_, 10);

    rclcpp::QoS map_qos(1);
    map_qos.transient_local().reliable();
    if (publish_map_) {
      map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(map_out_topic_, map_qos);
    }
    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      map_topic_, map_qos,
      std::bind(&ScanMatchingLocalizer::mapCallback, this, std::placeholders::_1));

    if (use_odom_) {
      odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, rclcpp::SensorDataQoS(),
        std::bind(&ScanMatchingLocalizer::odomCallback, this, std::placeholders::_1));
    }

    if (use_imu_) {
      imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, rclcpp::SensorDataQoS(),
        std::bind(&ScanMatchingLocalizer::imuCallback, this, std::placeholders::_1));
    }

    initpose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/initialpose", 10,
      std::bind(&ScanMatchingLocalizer::initPoseCallback, this, std::placeholders::_1));

    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic_, rclcpp::SensorDataQoS(),
      std::bind(&ScanMatchingLocalizer::scanCallback, this, std::placeholders::_1));

    if (use_dynamic_filter_) {
      opponent_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
        dynamic_obstacle_topic_, 10,
        std::bind(&ScanMatchingLocalizer::opponentCallback, this, std::placeholders::_1));
    }

    RCLCPP_INFO(
      get_logger(),
      "%s started: out=%s publish_tf=%d seed_from_tf=%d use_imu=%d use_odom=%d "
      "init=(%.2f, %.2f, %.2f)",
      node_name.c_str(), out_topic_.c_str(), publish_tf_, seed_from_tf_, use_imu_,
      use_odom_, pose_.x, pose_.y, pose_.theta);
  }

protected:
  // --- algorithm hooks implemented by the derived node ---
  virtual void buildMap(const Points & map_points) = 0;
  virtual Pose2D align(const Points & scan_base, const Pose2D & init) = 0;

  int maxIterations() const { return max_iterations_; }

private:
  void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    // Process the map once (it is static for this stack). The guard also stops
    // a feedback loop when we re-publish onto the same /map topic below.
    if (map_built_) return;

    const auto & info = msg->info;
    const double ox = info.origin.position.x;
    const double oy = info.origin.position.y;
    const double oyaw = tf2::getYaw(info.origin.orientation);
    const double c = std::cos(oyaw), s = std::sin(oyaw);
    const double res = info.resolution;

    Points pts;
    pts.reserve(info.width * info.height / 8 + 1);
    for (unsigned int j = 0; j < info.height; ++j) {
      for (unsigned int i = 0; i < info.width; ++i) {
        const int8_t v = msg->data[static_cast<std::size_t>(j) * info.width + i];
        if (v >= occ_threshold_) {
          const double lx = (i + 0.5) * res;
          const double ly = (j + 0.5) * res;
          pts.emplace_back(ox + c * lx - s * ly, oy + s * lx + c * ly);
        }
      }
    }
    if (pts.empty()) {
      RCLCPP_WARN(get_logger(), "Map has no cells >= threshold %d; ignoring", occ_threshold_);
      return;
    }
    buildMap(pts);
    map_ready_ = true;
    map_built_ = true;
    RCLCPP_INFO(get_logger(), "Map received: %zu occupied points", pts.size());

    // Re-publish the map once (latched) for visualization.
    if (publish_map_ && map_pub_) {
      msg->header.stamp = now();
      map_pub_->publish(*msg);
      RCLCPP_INFO(get_logger(), "Map re-published on %s for visualization", map_out_topic_.c_str());
    }
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    last_odom_.x = msg->pose.pose.position.x;
    last_odom_.y = msg->pose.pose.position.y;
    last_odom_.theta = tf2::getYaw(msg->pose.pose.orientation);
    have_odom_ = true;
  }

  // Integrate the IMU between scans. The gyro z-rate integrates to a heading
  // (the IMU is mounted yaw-only relative to base_link, so angular_velocity.z
  // is the base-link yaw rate); only per-scan deltas of imu_yaw_ are used, so a
  // constant gyro bias cancels. When running without wheel odometry, the body
  // linear acceleration is also integrated to a velocity (vx_, vy_) and a
  // translation (bx_, by_) accumulated since the last scan -- this is the
  // inter-scan dead-reckoning that seeds ICP. The accumulators are re-anchored
  // and the velocity re-set from the LiDAR motion every scan (finishScan), so
  // accelerometer bias / gravity leak cannot build up across scans.
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    const double t = rclcpp::Time(msg->header.stamp).seconds();
    if (have_imu_) {
      const double dt = t - last_imu_t_;
      if (dt > 0.0 && dt < 0.5) {  // ignore gaps / out-of-order stamps
        imu_yaw_ += imu_yaw_scale_ * msg->angular_velocity.z * dt;

        if (!use_odom_ && have_imu_ref_) {
          // Rotate the body-frame acceleration into the scan-anchored frame by
          // the heading change since the anchor, then integrate to velocity and
          // position (constant-acceleration step over dt).
          const double ah = wrapAngle(imu_yaw_ - imu_yaw_ref_);
          const double ca = std::cos(ah), sa = std::sin(ah);
          const double ax = msg->linear_acceleration.x;
          const double ay = msg->linear_acceleration.y;
          const double asx = ca * ax - sa * ay;
          const double asy = sa * ax + ca * ay;
          bx_ += vx_ * dt + 0.5 * asx * dt * dt;
          by_ += vy_ * dt + 0.5 * asy * dt * dt;
          vx_ += asx * dt;
          vy_ += asy * dt;
        }
      }
    }
    last_imu_t_ = t;
    have_imu_ = true;
  }

  void initPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  {
    pose_.x = msg->pose.pose.position.x;
    pose_.y = msg->pose.pose.position.y;
    pose_.theta = tf2::getYaw(msg->pose.pose.orientation);
    resetMotionRefs();  // restart the motion-delta chain from here
    pose_seeded_ = true;
    RCLCPP_INFO(
      get_logger(), "Initial pose set from /initialpose: (%.2f, %.2f, %.2f)",
      pose_.x, pose_.y, pose_.theta);
  }

  // In sim, gym_bridge owns map->base_link, so we can grab the start pose from
  // TF instead of needing a manual "2D Pose Estimate". Gated on seed_from_tf_
  // (only set on sim) so on real HW we never read back our own published TF.
  void trySeedFromTf()
  {
    if (pose_seeded_ || !seed_from_tf_) return;
    try {
      const auto t = tf_buffer_->lookupTransform(map_frame_, base_frame_, tf2::TimePointZero);
      pose_.x = t.transform.translation.x;
      pose_.y = t.transform.translation.y;
      pose_.theta = tf2::getYaw(t.transform.rotation);
      pose_seeded_ = true;
      resetMotionRefs();
      RCLCPP_INFO(
        get_logger(), "Seeded pose from TF %s->%s: (%.2f, %.2f, %.2f)",
        map_frame_.c_str(), base_frame_.c_str(), pose_.x, pose_.y, pose_.theta);
    } catch (const std::exception &) {
      // TF not up yet; retry next scan.
    }
  }

  void ensureLaserExtrinsic(const std::string & scan_frame)
  {
    if (have_laser_tf_ || scan_frame.empty()) return;
    try {
      const auto t = tf_buffer_->lookupTransform(base_frame_, scan_frame, tf2::TimePointZero);
      laser_x_ = t.transform.translation.x;
      laser_y_ = t.transform.translation.y;
      laser_yaw_ = tf2::getYaw(t.transform.rotation);
      have_laser_tf_ = true;
      RCLCPP_INFO(
        get_logger(), "%s->%s extrinsic from TF: (%.3f, %.3f, %.3f rad)",
        base_frame_.c_str(), scan_frame.c_str(), laser_x_, laser_y_, laser_yaw_);
    } catch (const std::exception &) {
      // Fall back to the laser_* params; retry next scan.
    }
  }

  // Forget the running motion references so the next scan starts a fresh
  // prediction chain (called after a pose jump from /initialpose or TF seed).
  void resetMotionRefs()
  {
    have_odom_ref_ = false;
    have_imu_ref_ = false;
    have_scan_t_ = false;
    vx_ = vy_ = 0.0;
    bx_ = by_ = 0.0;
  }

  // Predict the pose at the current scan from the previous pose `from`. The
  // body-frame translation comes from wheel odometry (use_odom), or -- when
  // use_odom is false -- from the IMU dead-reckoning accumulated since the last
  // scan (bx_, by_; see imuCallback). The heading delta is taken from the
  // integrated IMU gyro when available, overriding the translation source's yaw.
  // Read-only: the running references are advanced in finishScan() once ICP has
  // produced the corrected pose.
  Pose2D predictMotion(const Pose2D & from)
  {
    Pose2D d;  // body-frame delta to apply to `from`; identity by default
    bool have_delta = false;

    if (use_odom_ && have_odom_ && have_odom_ref_) {
      d = relative(odom_ref_, last_odom_);
      have_delta = true;
    } else if (!use_odom_ && have_imu_ref_) {
      d.x = bx_;  // IMU dead-reckoned translation since the last scan
      d.y = by_;
      have_delta = true;
    }

    if (use_imu_ && have_imu_ && have_imu_ref_) {
      d.theta = wrapAngle(imu_yaw_ - imu_yaw_ref_);
      have_delta = true;
    }

    return have_delta ? compose(from, d) : from;
  }

  // Advance the motion references after ICP has corrected the pose. Anchors the
  // wheel-odom / IMU-heading references at this scan and, for the odom-free
  // mode, re-sets the body velocity from the LiDAR-observed motion (m / dt) and
  // zeroes the translation accumulator -- so the next inter-scan integration
  // starts from a LiDAR-anchored velocity and bias cannot accumulate.
  void finishScan(const Pose2D & prev_pose, double dt)
  {
    if (use_odom_ && have_odom_) {
      odom_ref_ = last_odom_;
      have_odom_ref_ = true;
    }
    if (use_imu_ && have_imu_) {
      imu_yaw_ref_ = imu_yaw_;
      have_imu_ref_ = true;
    }

    bx_ = 0.0;
    by_ = 0.0;
    if (!use_odom_ && dt > 1e-3) {
      const Pose2D m = relative(prev_pose, pose_);   // body-frame motion this scan
      const double ct = std::cos(m.theta), st = std::sin(m.theta);
      // Express the velocity in the new body frame: R(-m.theta) * (m.x, m.y) / dt.
      vx_ = (ct * m.x + st * m.y) / dt;
      vy_ = (-st * m.x + ct * m.y) / dt;
    }
  }

  // Latest perception-tracked opponent centre, in the map frame.
  void opponentCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg)
  {
    opp_x_ = msg->point.x;
    opp_y_ = msg->point.y;
    opp_t_ = rclcpp::Time(msg->header.stamp).seconds();
    have_opp_ = true;
  }

  // Remove base-frame scan points that fall within dynamic_filter_radius_ of the
  // tracked opponent. The opponent is in the map frame, so we project it into the
  // base frame with the predicted pose. Skips when the detection is missing/stale.
  void filterDynamicPoints(Points & pts, const Pose2D & pred, const rclcpp::Time & scan_t)
  {
    if (!have_opp_) return;
    const double age = scan_t.seconds() - opp_t_;
    if (age < 0.0 || age > dynamic_filter_timeout_) return;   // stale → keep all points

    // map -> base: rotate the (opponent - pose) offset by -pred.theta.
    const double dx = opp_x_ - pred.x, dy = opp_y_ - pred.y;
    const double ct = std::cos(pred.theta), st = std::sin(pred.theta);
    const double ox =  ct * dx + st * dy;   // opponent in base frame
    const double oy = -st * dx + ct * dy;
    const double r2 = dynamic_filter_radius_ * dynamic_filter_radius_;

    std::size_t kept = 0;
    for (const auto & p : pts) {
      const double ex = p.x() - ox, ey = p.y() - oy;
      if (ex * ex + ey * ey > r2) pts[kept++] = p;
    }
    const std::size_t removed = pts.size() - kept;
    pts.resize(kept);
    if (debug_timing_ && removed > 0) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "dynamic filter: removed %zu pts around opponent (r=%.2f m)",
        removed, dynamic_filter_radius_);
    }
  }

  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    ensureLaserExtrinsic(msg->header.frame_id);
    trySeedFromTf();

    // Convert beams to points in the base_link frame, dropping invalid returns.
    Points pts;
    pts.reserve(msg->ranges.size());
    const double cL = std::cos(laser_yaw_), sL = std::sin(laser_yaw_);
    double a = msg->angle_min;
    for (std::size_t i = 0; i < msg->ranges.size(); ++i, a += msg->angle_increment) {
      const double r = msg->ranges[i];
      if (!std::isfinite(r) || r < min_range_ || r < msg->range_min ||
        r > max_range_ || r > msg->range_max)
      {
        continue;
      }
      const double lx = r * std::cos(a), ly = r * std::sin(a);
      pts.emplace_back(laser_x_ + cL * lx - sL * ly, laser_y_ + sL * lx + cL * ly);
    }

    // Motion prediction since the last scan (translation from wheel odometry or
    // IMU dead-reckoning; heading from the IMU gyro when available).
    const Pose2D prev_pose = pose_;
    const Pose2D pred = predictMotion(prev_pose);

    // Drop scan points around the tracked opponent (dynamic-object rejection)
    // before registration. No-op unless use_dynamic_filter and a fresh detection.
    if (use_dynamic_filter_) {
      filterDynamicPoints(pts, pred, rclcpp::Time(msg->header.stamp));
    }

    if (map_ready_ && static_cast<int>(pts.size()) >= min_scan_points_) {
      const auto t0 = std::chrono::steady_clock::now();
      const Pose2D icp_pose = align(pts, pred);
      const auto t1 = std::chrono::steady_clock::now();

      // Fitness gate: decide whether to USE the LiDAR (ICP) pose or coast on the
      // motion prediction. Done here so a bad scan match never corrupts pose_.
      const bool fit_ok =
        (last_inlier_ratio_ >= fitness_min_inlier_ratio_) &&
        (last_mean_resid_ <= fitness_max_resid_);
      const bool use_icp = !fitness_gate_enable_ || fit_ok;
      pose_ = use_icp ? icp_pose : pred;

      if (debug_timing_) {
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        // Throttled to 1 Hz so a 40 Hz scan stream does not flood the console.
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "align: %.2f ms  (%d iters, %zu scan pts, %d corr, "
          "inlier %.0f%%, resid %.3f m) -> %s",
          ms, last_iterations_, pts.size(), last_correspondences_,
          100.0 * last_inlier_ratio_, last_mean_resid_,
          use_icp ? "ICP" : "DEAD-RECKON");
      }
      if (fitness_gate_enable_ && !fit_ok) {
        // Throttled so a sustained bad-fit stretch warns ~1 Hz, not per scan.
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "ICP fit rejected (inlier %.0f%% < %.0f%% or resid %.3f > %.3f m) "
          "-> coasting on motion prediction",
          100.0 * last_inlier_ratio_, 100.0 * fitness_min_inlier_ratio_,
          last_mean_resid_, fitness_max_resid_);
      }
    } else {
      pose_ = pred;
    }

    // Inter-scan interval, used to re-set the dead-reckoning velocity from the
    // LiDAR-observed motion.
    const double t = rclcpp::Time(msg->header.stamp).seconds();
    const double dt_scan = have_scan_t_ ? (t - last_scan_t_) : 0.0;
    last_scan_t_ = t;
    have_scan_t_ = true;

    finishScan(prev_pose, dt_scan);

    publishResult(msg->header.stamp);
  }

  void publishResult(const rclcpp::Time & stamp)
  {
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, pose_.theta);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = map_frame_;
    odom.child_frame_id = base_frame_;
    odom.pose.pose.position.x = pose_.x;
    odom.pose.pose.position.y = pose_.y;
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();
    // Diagonal pose covariance (x, y, z, roll, pitch, yaw); huge for unused 2D axes.
    odom.pose.covariance[0] = 0.025;
    odom.pose.covariance[7] = 0.025;
    odom.pose.covariance[14] = 1e6;
    odom.pose.covariance[21] = 1e6;
    odom.pose.covariance[28] = 1e6;
    odom.pose.covariance[35] = 0.05;
    odom_pub_->publish(odom);

    if (publish_tf_) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp = stamp;
      tf.header.frame_id = map_frame_;
      tf.child_frame_id = base_frame_;
      tf.transform.translation.x = pose_.x;
      tf.transform.translation.y = pose_.y;
      tf.transform.translation.z = 0.0;
      tf.transform.rotation = odom.pose.pose.orientation;
      tf_broadcaster_->sendTransform(tf);
    }
  }

protected:
  Pose2D pose_;
  double laser_x_ = 0.27, laser_y_ = 0.0, laser_yaw_ = 0.0;
  int last_iterations_ = 0;  // iterations used by the most recent align(); set by derived
  int last_correspondences_ = 0;  // correspondence pairs in the last align() iteration; set by derived
  // Registration fitness of the most recent align(), evaluated at a FIXED tight
  // distance (independent of the adaptive corr gate) so it actually reflects how
  // well the scan sits on the map. Set by derived align(). A high correspondence
  // COUNT can coexist with a bad fit (every point finds *some* wall), so these —
  // not last_correspondences_ — are the localization health signal.
  double last_inlier_ratio_ = 0.0;  // fraction of scan points within the fitness distance
  double last_mean_resid_ = 0.0;    // RMS residual over those inliers [m]

private:
  std::string scan_topic_, odom_topic_, imu_topic_, map_topic_;
  std::string map_frame_, base_frame_, out_topic_;
  bool publish_tf_ = true;
  bool seed_from_tf_ = false;
  bool use_imu_ = true;
  bool use_odom_ = true;
  double imu_yaw_scale_ = 1.0;
  bool publish_map_ = true;
  bool debug_timing_ = true;
  std::string map_out_topic_ = "/map";
  double min_range_ = 0.1, max_range_ = 15.0;
  int max_iterations_ = 20;
  int min_scan_points_ = 30;
  int occ_threshold_ = 65;
  bool fitness_gate_enable_ = true;
  double fitness_min_inlier_ratio_ = 0.55;
  double fitness_max_resid_ = 0.12;
  bool use_dynamic_filter_ = false;
  double dynamic_filter_radius_ = 0.60;
  double dynamic_filter_timeout_ = 0.30;
  std::string dynamic_obstacle_topic_ = "/local_planning/opponent";
  double opp_x_ = 0.0, opp_y_ = 0.0;
  double opp_t_ = 0.0;   // detection stamp [s] (clock-type-agnostic)
  bool have_opp_ = false;

  bool map_ready_ = false;
  bool map_built_ = false;
  bool have_odom_ = false, have_odom_ref_ = false;
  bool have_imu_ = false, have_imu_ref_ = false;
  bool have_scan_t_ = false;
  bool have_laser_tf_ = false;
  bool pose_seeded_ = false;
  Pose2D last_odom_, odom_ref_;
  double imu_yaw_ = 0.0, imu_yaw_ref_ = 0.0, last_imu_t_ = 0.0;
  // IMU dead-reckoning state (odom-free mode): body velocity and the translation
  // accumulated since the last scan, plus the last scan timestamp.
  double vx_ = 0.0, vy_ = 0.0, bx_ = 0.0, by_ = 0.0, last_scan_t_ = 0.0;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initpose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr opponent_sub_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace scan_matching_localization
