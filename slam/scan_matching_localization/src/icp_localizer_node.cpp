// kiss-icp-style scan-to-map localizer.
//
// Point-to-point ICP between the (voxel-downsampled) current scan and the
// static map point cloud. Two ideas borrowed from KISS-ICP keep it robust on a
// 2D LiDAR without any tuning per map:
//   * voxel downsampling of the scan, so cost is independent of beam density;
//   * an adaptive correspondence threshold that starts loose (to tolerate the
//     wheel-odom prediction error) and shrinks toward a tight value as the
//     registration converges.
// The motion model (constant body-frame velocity from /vesc/odom) is handled by
// the base class, which hands us the predicted pose as the ICP seed.
#include <algorithm>

#include "scan_matching_localization/scan_matching_localizer.hpp"
#include "scan_matching_localization/spatial_grid.hpp"

using namespace scan_matching_localization;

class IcpLocalizer : public ScanMatchingLocalizer
{
public:
  IcpLocalizer()
  : ScanMatchingLocalizer("icp_localizer", "/icp/pose/odom")
  {
    voxel_size_ = declare_parameter<double>("voxel_size", 0.1);
    max_corr_dist_ = declare_parameter<double>("max_correspondence_distance", 1.0);
    min_corr_dist_ = declare_parameter<double>("min_correspondence_distance", 0.15);
    corr_decay_ = declare_parameter<double>("correspondence_decay", 0.85);
    nn_cell_size_ = declare_parameter<double>("nn_cell_size", 0.5);
    trans_eps_ = declare_parameter<double>("transform_epsilon", 1e-3);
    rot_eps_ = declare_parameter<double>("rotation_epsilon", 1e-4);
    min_corr_pairs_ = declare_parameter<int>("min_correspondence_pairs", 10);

    // Point-to-line (PL-ICP) variant: minimize the residual along the map normal
    // instead of point-to-point. Faster/sharper on walls; falls back to
    // point-to-point per correspondence where the map normal is unreliable.
    use_point_to_plane_ = declare_parameter<bool>("use_point_to_plane", false);
    normal_radius_ = declare_parameter<double>("normal_radius", 0.30);
    normal_min_neighbors_ = declare_parameter<int>("normal_min_neighbors", 6);
    normal_linearity_ = declare_parameter<double>("normal_linearity", 0.30);
  }

protected:
  void buildMap(const Points & map_points) override
  {
    grid_.build(
      map_points, nn_cell_size_, use_point_to_plane_, normal_radius_,
      normal_min_neighbors_, normal_linearity_);
  }

  Pose2D align(const Points & scan_base, const Pose2D & init) override
  {
    if (grid_.empty()) return init;
    const Points src = voxelDownsample(scan_base, voxel_size_);

    Pose2D T = init;
    double corr = max_corr_dist_;

    for (int iter = 0; iter < maxIterations(); ++iter) {
      last_iterations_ = iter + 1;
      const double c = std::cos(T.theta), s = std::sin(T.theta);

      if (use_point_to_plane_) {
        if (alignPointToLineStep(src, c, s, corr, T)) break;
        corr = std::max(min_corr_dist_, corr * corr_decay_);
        continue;
      }

      Points P, Q;  // current world points and their map correspondences
      P.reserve(src.size());
      Q.reserve(src.size());
      Eigen::Vector2d mu_p = Eigen::Vector2d::Zero();
      Eigen::Vector2d mu_q = Eigen::Vector2d::Zero();

      for (const auto & sp : src) {
        const Eigen::Vector2d wp(
          T.x + c * sp.x() - s * sp.y(),
          T.y + s * sp.x() + c * sp.y());
        double d2;
        const int idx = grid_.nearest(wp, corr, d2);
        if (idx < 0) continue;
        const Eigen::Vector2d & qp = grid_.point(idx);
        P.push_back(wp);
        Q.push_back(qp);
        mu_p += wp;
        mu_q += qp;
      }

      if (static_cast<int>(P.size()) < min_corr_pairs_) break;

      const double n = static_cast<double>(P.size());
      mu_p /= n;
      mu_q /= n;

      // Best rigid 2D transform mapping P onto Q (Umeyama / Kabsch).
      Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
      for (std::size_t i = 0; i < P.size(); ++i) {
        H += (P[i] - mu_p) * (Q[i] - mu_q).transpose();
      }
      Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Matrix2d R = svd.matrixV() * svd.matrixU().transpose();
      if (R.determinant() < 0.0) {
        Eigen::Matrix2d V = svd.matrixV();
        V.col(1) *= -1.0;
        R = V * svd.matrixU().transpose();
      }
      const Eigen::Vector2d t = mu_q - R * mu_p;

      // Compose the increment dT = (R, t) onto T:  T' = dT * T.
      const double dtheta = std::atan2(R(1, 0), R(0, 0));
      const Pose2D Told = T;
      T.x = R(0, 0) * Told.x + R(0, 1) * Told.y + t.x();
      T.y = R(1, 0) * Told.x + R(1, 1) * Told.y + t.y();
      T.theta = wrapAngle(Told.theta + dtheta);

      // Tighten the gate as we converge (adaptive threshold).
      corr = std::max(min_corr_dist_, corr * corr_decay_);

      const double dtrans = std::hypot(T.x - Told.x, T.y - Told.y);
      if (dtrans < trans_eps_ && std::abs(dtheta) < rot_eps_) break;
    }
    return T;
  }

  // One Gauss-Newton step minimizing the point-to-line residual e = n^T (p - q)
  // over the pose (x, y, theta). Where the map normal is unreliable the point
  // contributes a point-to-point residual instead, so no constraint is lost.
  // Updates T in place; returns true if the step converged or there were too
  // few correspondences (i.e. the caller should stop iterating).
  bool alignPointToLineStep(
    const Points & src, double c, double s, double corr, Pose2D & T)
  {
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    int pairs = 0;

    for (const auto & sp : src) {
      const Eigen::Vector2d wp(
        T.x + c * sp.x() - s * sp.y(),
        T.y + s * sp.x() + c * sp.y());
      double d2;
      const int idx = grid_.nearest(wp, corr, d2);
      if (idx < 0) continue;
      ++pairs;
      const Eigen::Vector2d & qp = grid_.point(idx);
      // d(wp)/d(x,y,theta) = [[1,0,-(wp.y-T.y)], [0,1,(wp.x-T.x)]]
      const double dx = wp.x() - T.x, dy = wp.y() - T.y;

      if (grid_.normalValid(idx)) {
        const Eigen::Vector2d & n = grid_.normal(idx);
        Eigen::Vector3d jr;            // jr = n^T J  (1x3)
        jr << n.x(), n.y(), -n.x() * dy + n.y() * dx;
        const double e = n.dot(wp - qp);
        H += jr * jr.transpose();
        g += jr * e;
      } else {
        Eigen::Matrix<double, 2, 3> J;
        J << 1.0, 0.0, -dy,
             0.0, 1.0,  dx;
        const Eigen::Vector2d e = wp - qp;
        H += J.transpose() * J;
        g += J.transpose() * e;
      }
    }

    if (pairs < min_corr_pairs_) return true;

    H.diagonal().array() += 1e-9;  // keep H invertible when weakly constrained
    const Eigen::Vector3d delta = H.ldlt().solve(-g);
    T.x += delta(0);
    T.y += delta(1);
    T.theta = wrapAngle(T.theta + delta(2));
    return std::hypot(delta(0), delta(1)) < trans_eps_ && std::abs(delta(2)) < rot_eps_;
  }

private:
  NearestGrid grid_;
  double voxel_size_;
  double max_corr_dist_;
  double min_corr_dist_;
  double corr_decay_;
  double nn_cell_size_;
  double trans_eps_;
  double rot_eps_;
  int min_corr_pairs_;
  bool use_point_to_plane_;
  double normal_radius_;
  int normal_min_neighbors_;
  double normal_linearity_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<IcpLocalizer>());
  rclcpp::shutdown();
  return 0;
}
