// Base node for scan-to-map 2D LiDAR localization.
//
// I/O contract (identical to the particle_filter / cartographer modes so it
// drops straight into stack_master/launch/middle_level.launch.xml):
//   subscribes : /scan        (sensor_msgs/LaserScan, frame = laser)
//                /vesc/odom    (nav_msgs/Odometry,    translation prediction)
//                /vesc/sensors/imu/raw (sensor_msgs/Imu, heading prediction)
//                /map          (nav_msgs/OccupancyGrid, transient-local latch)
//                /initialpose  (geometry_msgs/PoseWithCovarianceStamped, RViz)
//   publishes  : <out>/pose/odom (nav_msgs/Odometry, pose in map frame) -> EKF
//                map -> base_link TF (only when publish_tf:=true, i.e. real HW)
//
// Pipeline per scan: predict the new pose (translation from wheel odometry,
// heading from the integrated IMU gyro when available), then refine it by
// registering the scan against the map with the algorithm supplied by the
// derived class (ICP or NDT).
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

    // Motion model: translation always from /vesc/odom; heading from the IMU
    // gyro when use_imu and IMU data are available, else from odom yaw.
    use_imu_ = declare_parameter<bool>("use_imu", true);
    imu_yaw_scale_ = declare_parameter<double>("imu_yaw_scale", 1.0);

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

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, rclcpp::SensorDataQoS(),
      std::bind(&ScanMatchingLocalizer::odomCallback, this, std::placeholders::_1));

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

    RCLCPP_INFO(
      get_logger(),
      "%s started: out=%s publish_tf=%d seed_from_tf=%d use_imu=%d init=(%.2f, %.2f, %.2f)",
      node_name.c_str(), out_topic_.c_str(), publish_tf_, seed_from_tf_, use_imu_,
      pose_.x, pose_.y, pose_.theta);
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

  // Integrate the gyro yaw rate. The IMU is mounted yaw-only relative to
  // base_link (z aligned), so angular_velocity.z is the base-link yaw rate.
  // Only per-scan deltas of imu_yaw_ are used, so a constant gyro bias cancels.
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    const double t = rclcpp::Time(msg->header.stamp).seconds();
    if (have_imu_) {
      const double dt = t - last_imu_t_;
      if (dt > 0.0 && dt < 0.5) {  // ignore gaps / out-of-order stamps
        imu_yaw_ += imu_yaw_scale_ * msg->angular_velocity.z * dt;
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
    have_odom_ref_ = false;  // restart the motion-delta chain from here
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
      have_odom_ref_ = false;
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

    // Motion prediction since the last scan: translation from wheel odometry,
    // heading from the integrated IMU gyro when available (else odom yaw).
    Pose2D pred = pose_;
    if (have_odom_) {
      if (have_odom_ref_) {
        Pose2D d = relative(odom_ref_, last_odom_);
        if (use_imu_ && have_imu_) {
          d.theta = wrapAngle(imu_yaw_ - imu_yaw_ref_);
        }
        pred = compose(pose_, d);
      }
      odom_ref_ = last_odom_;
      imu_yaw_ref_ = imu_yaw_;
      have_odom_ref_ = true;
    }

    if (map_ready_ && static_cast<int>(pts.size()) >= min_scan_points_) {
      const auto t0 = std::chrono::steady_clock::now();
      pose_ = align(pts, pred);
      const auto t1 = std::chrono::steady_clock::now();
      if (debug_timing_) {
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        // Throttled to 1 Hz so a 40 Hz scan stream does not flood the console.
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "align: %.2f ms  (%d iters, %zu scan pts)", ms, last_iterations_, pts.size());
      }
    } else {
      pose_ = pred;
    }

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

private:
  std::string scan_topic_, odom_topic_, imu_topic_, map_topic_;
  std::string map_frame_, base_frame_, out_topic_;
  bool publish_tf_ = true;
  bool seed_from_tf_ = false;
  bool use_imu_ = true;
  double imu_yaw_scale_ = 1.0;
  bool publish_map_ = true;
  bool debug_timing_ = true;
  std::string map_out_topic_ = "/map";
  double min_range_ = 0.1, max_range_ = 15.0;
  int max_iterations_ = 20;
  int min_scan_points_ = 30;
  int occ_threshold_ = 65;

  bool map_ready_ = false;
  bool map_built_ = false;
  bool have_odom_ = false, have_odom_ref_ = false;
  bool have_imu_ = false;
  bool have_laser_tf_ = false;
  bool pose_seeded_ = false;
  Pose2D last_odom_, odom_ref_;
  double imu_yaw_ = 0.0, imu_yaw_ref_ = 0.0, last_imu_t_ = 0.0;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initpose_sub_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace scan_matching_localization
