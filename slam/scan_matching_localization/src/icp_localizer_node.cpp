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
  }

protected:
  void buildMap(const Points & map_points) override
  {
    grid_.build(map_points, nn_cell_size_);
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
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<IcpLocalizer>());
  rclcpp::shutdown();
  return 0;
}
