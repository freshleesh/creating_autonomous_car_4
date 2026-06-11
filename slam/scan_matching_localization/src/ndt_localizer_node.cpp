// NDT (Normal Distributions Transform) scan-to-map localizer.
//
// The map is discretized into a grid of 2D Gaussians (mean + covariance per
// cell). For each scan we maximize the sum of cell likelihoods over the pose
// (tx, ty, theta) with Gauss-Newton: a positive-semidefinite Hessian
// approximation (Biber & Strasser 2003 / Magnusson 2009) that drops the
// second-order term, which keeps every step a descent direction without a line
// search. The base class supplies the wheel-odom-predicted pose as the seed.
#include "scan_matching_localization/scan_matching_localizer.hpp"
#include "scan_matching_localization/spatial_grid.hpp"

using namespace scan_matching_localization;

class NdtLocalizer : public ScanMatchingLocalizer
{
public:
  NdtLocalizer()
  : ScanMatchingLocalizer("ndt_localizer", "/ndt/pose/odom")
  {
    voxel_size_ = declare_parameter<double>("voxel_size", 0.1);
    ndt_resolution_ = declare_parameter<double>("ndt_resolution", 1.0);
    min_pts_per_cell_ = declare_parameter<int>("min_points_per_cell", 5);
    min_cell_variance_ = declare_parameter<double>("min_cell_variance", 0.06);
    max_mahalanobis_ = declare_parameter<double>("max_mahalanobis_sq", 30.0);
    step_size_ = declare_parameter<double>("step_size", 1.0);
    trans_eps_ = declare_parameter<double>("transform_epsilon", 1e-3);
    rot_eps_ = declare_parameter<double>("rotation_epsilon", 1e-4);
    hessian_reg_ = declare_parameter<double>("hessian_regularization", 1e-6);
  }

protected:
  void buildMap(const Points & map_points) override
  {
    grid_.build(map_points, ndt_resolution_, min_pts_per_cell_, min_cell_variance_);
    RCLCPP_INFO(get_logger(), "NDT grid built: %zu valid cells", grid_.size());
  }

  Pose2D align(const Points & scan_base, const Pose2D & init) override
  {
    if (grid_.empty()) return init;
    const Points src = voxelDownsample(scan_base, voxel_size_);

    Pose2D T = init;
    for (int iter = 0; iter < maxIterations(); ++iter) {
      last_iterations_ = iter + 1;
      const double c = std::cos(T.theta), s = std::sin(T.theta);

      Eigen::Vector3d g = Eigen::Vector3d::Zero();   // gradient of -score
      Eigen::Matrix3d Hh = Eigen::Matrix3d::Zero();  // Gauss-Newton Hessian

      for (const auto & p : src) {
        const Eigen::Vector2d tp(
          T.x + c * p.x() - s * p.y(),
          T.y + s * p.x() + c * p.y());
        const NdtCell * cell = grid_.at(tp);
        if (cell == nullptr) continue;

        const Eigen::Vector2d xv = tp - cell->mean;
        const Eigen::Vector2d a = cell->inv_cov * xv;  // C^-1 x
        const double e = xv.dot(a);
        if (e > max_mahalanobis_) continue;  // negligible likelihood, skip the tail
        const double q = std::exp(-0.5 * e);

        // d(tp)/d(tx, ty, theta)
        Eigen::Matrix<double, 2, 3> J;
        J(0, 0) = 1.0; J(0, 1) = 0.0; J(0, 2) = -s * p.x() - c * p.y();
        J(1, 0) = 0.0; J(1, 1) = 1.0; J(1, 2) = c * p.x() - s * p.y();

        g += q * (J.transpose() * a);
        Hh += q * (J.transpose() * cell->inv_cov * J);
      }

      Hh += hessian_reg_ * Eigen::Matrix3d::Identity();
      // Newton step that minimizes -score (maximizes the NDT likelihood).
      const Eigen::Vector3d dx = Hh.ldlt().solve(-g);

      T.x += step_size_ * dx(0);
      T.y += step_size_ * dx(1);
      T.theta = wrapAngle(T.theta + step_size_ * dx(2));

      if (std::hypot(dx(0), dx(1)) < trans_eps_ && std::abs(dx(2)) < rot_eps_) break;
    }
    return T;
  }

private:
  NdtGrid grid_;
  double voxel_size_;
  double ndt_resolution_;
  int min_pts_per_cell_;
  double min_cell_variance_;
  double max_mahalanobis_;
  double step_size_;
  double trans_eps_;
  double rot_eps_;
  double hessian_reg_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<NdtLocalizer>());
  rclcpp::shutdown();
  return 0;
}
