// Dependency-free 2D spatial structures used by the scan-to-map localizers.
//   * NearestGrid : uniform-grid nearest-neighbour lookup for ICP correspondences.
//   * NdtGrid     : per-cell Gaussian (mean + inverse covariance) for NDT.
// A uniform grid is enough here: queries are always bounded by a max
// correspondence / cell distance, so a fixed neighbourhood sweep is O(1) and
// avoids pulling in PCL / nanoflann / Open3D (none are available on this box).
#pragma once

#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

namespace scan_matching_localization
{

struct CellKey
{
  int i;
  int j;
  bool operator==(const CellKey & o) const { return i == o.i && j == o.j; }
};

struct CellKeyHash
{
  std::size_t operator()(const CellKey & k) const
  {
    // Pack two 32-bit ints into 64 bits, then hash.
    uint64_t a = static_cast<uint32_t>(k.i);
    uint64_t b = static_cast<uint32_t>(k.j);
    return std::hash<uint64_t>()((a << 32) ^ b);
  }
};

inline CellKey cellOf(const Eigen::Vector2d & p, double inv_cell)
{
  return {static_cast<int>(std::floor(p.x() * inv_cell)),
          static_cast<int>(std::floor(p.y() * inv_cell))};
}

// Keep at most one point per voxel — cheap downsampling so every scan costs
// roughly the same regardless of how many beams land on a near wall.
inline std::vector<Eigen::Vector2d> voxelDownsample(
  const std::vector<Eigen::Vector2d> & pts, double voxel)
{
  if (voxel <= 0.0) return pts;
  const double inv = 1.0 / voxel;
  std::unordered_map<CellKey, Eigen::Vector2d, CellKeyHash> picked;
  picked.reserve(pts.size());
  std::vector<Eigen::Vector2d> out;
  out.reserve(pts.size());
  for (const auto & p : pts) {
    if (picked.emplace(cellOf(p, inv), p).second) out.push_back(p);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Nearest-neighbour grid (ICP)
// ---------------------------------------------------------------------------
class NearestGrid
{
public:
  // with_normals: precompute a per-point surface normal (for point-to-line ICP).
  //   The map is static, so this one-time cost keeps the per-scan loop cheap.
  //   A normal is the smallest-eigenvalue eigenvector of the local neighbourhood
  //   covariance; it is marked invalid where the neighbourhood is too sparse or
  //   too isotropic (blob, not a line) to give a reliable direction.
  void build(
    const std::vector<Eigen::Vector2d> & pts, double cell,
    bool with_normals = false, double normal_radius = 0.3,
    int normal_min_neighbors = 6, double normal_linearity = 0.3)
  {
    cell_ = cell;
    inv_ = 1.0 / cell;
    pts_ = pts;
    grid_.clear();
    grid_.reserve(pts.size());
    for (std::size_t idx = 0; idx < pts_.size(); ++idx) {
      grid_[cellOf(pts_[idx], inv_)].push_back(static_cast<int>(idx));
    }
    if (with_normals) {
      computeNormals(normal_radius, normal_min_neighbors, normal_linearity);
    }
  }

  // Index of nearest stored point within max_dist, or -1. best_d2 = squared dist.
  int nearest(const Eigen::Vector2d & p, double max_dist, double & best_d2) const
  {
    const int ci = static_cast<int>(std::floor(p.x() * inv_));
    const int cj = static_cast<int>(std::floor(p.y() * inv_));
    const int reach = static_cast<int>(std::ceil(max_dist * inv_));
    best_d2 = max_dist * max_dist;
    int best = -1;
    for (int di = -reach; di <= reach; ++di) {
      for (int dj = -reach; dj <= reach; ++dj) {
        auto it = grid_.find({ci + di, cj + dj});
        if (it == grid_.end()) continue;
        for (int idx : it->second) {
          const double d2 = (pts_[idx] - p).squaredNorm();
          if (d2 < best_d2) {
            best_d2 = d2;
            best = idx;
          }
        }
      }
    }
    return best;
  }

  const Eigen::Vector2d & point(int idx) const { return pts_[idx]; }
  bool empty() const { return pts_.empty(); }

  // Per-point surface normal (unit, sign arbitrary). Only valid where
  // normalValid(idx); call build(..., with_normals=true) first.
  const Eigen::Vector2d & normal(int idx) const { return normals_[idx]; }
  bool normalValid(int idx) const
  {
    return idx >= 0 && static_cast<std::size_t>(idx) < normal_valid_.size() &&
           normal_valid_[idx];
  }

private:
  // Estimate each point's normal from neighbours within normal_radius via PCA.
  void computeNormals(double radius, int min_neighbors, double linearity)
  {
    normals_.assign(pts_.size(), Eigen::Vector2d::Zero());
    normal_valid_.assign(pts_.size(), false);
    const double r2 = radius * radius;
    const int reach = std::max(1, static_cast<int>(std::ceil(radius * inv_)));

    std::vector<Eigen::Vector2d> nb;
    for (std::size_t idx = 0; idx < pts_.size(); ++idx) {
      const Eigen::Vector2d & p = pts_[idx];
      const int ci = static_cast<int>(std::floor(p.x() * inv_));
      const int cj = static_cast<int>(std::floor(p.y() * inv_));

      nb.clear();
      for (int di = -reach; di <= reach; ++di) {
        for (int dj = -reach; dj <= reach; ++dj) {
          auto it = grid_.find({ci + di, cj + dj});
          if (it == grid_.end()) continue;
          for (int k : it->second) {
            if ((pts_[k] - p).squaredNorm() <= r2) nb.push_back(pts_[k]);
          }
        }
      }
      if (static_cast<int>(nb.size()) < min_neighbors) continue;

      Eigen::Vector2d mean = Eigen::Vector2d::Zero();
      for (const auto & q : nb) mean += q;
      mean /= static_cast<double>(nb.size());

      Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
      for (const auto & q : nb) {
        const Eigen::Vector2d d = q - mean;
        cov += d * d.transpose();
      }
      cov /= static_cast<double>(nb.size());

      // Eigenvalues ascending: ev(0) across the line (normal), ev(1) along it.
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
      const Eigen::Vector2d ev = es.eigenvalues();
      if (ev(1) <= 0.0) continue;
      if (ev(0) / ev(1) > linearity) continue;  // too isotropic -> ambiguous normal
      normals_[idx] = es.eigenvectors().col(0).normalized();
      normal_valid_[idx] = true;
    }
  }

  double cell_ = 0.5;
  double inv_ = 2.0;
  std::vector<Eigen::Vector2d> pts_;
  std::unordered_map<CellKey, std::vector<int>, CellKeyHash> grid_;
  std::vector<Eigen::Vector2d> normals_;
  std::vector<bool> normal_valid_;
};

// ---------------------------------------------------------------------------
// NDT cell grid
// ---------------------------------------------------------------------------
struct NdtCell
{
  Eigen::Vector2d mean = Eigen::Vector2d::Zero();
  Eigen::Matrix2d inv_cov = Eigen::Matrix2d::Identity();
  bool valid = false;
};

class NdtGrid
{
public:
  // min_var: absolute lower bound on each covariance eigenvalue [m^2]. For a
  // thin wall segment the normal-direction variance is ~0, which would make the
  // Gaussian razor-thin and collapse the convergence basin to a couple of cm.
  // Flooring it to a sensor/discretization-scale value widens the basin to be
  // comparable to the cell size without moving the likelihood peak (the optimum
  // stays at the cell mean regardless of the floor).
  void build(const std::vector<Eigen::Vector2d> & pts, double cell, int min_pts, double min_var)
  {
    cell_ = cell;
    inv_ = 1.0 / cell;
    cells_.clear();

    std::unordered_map<CellKey, std::vector<Eigen::Vector2d>, CellKeyHash> acc;
    acc.reserve(pts.size() / 4 + 1);
    for (const auto & p : pts) acc[cellOf(p, inv_)].push_back(p);

    cells_.reserve(acc.size());
    for (auto & kv : acc) {
      const auto & cp = kv.second;
      if (static_cast<int>(cp.size()) < min_pts) continue;

      Eigen::Vector2d mean = Eigen::Vector2d::Zero();
      for (const auto & p : cp) mean += p;
      mean /= static_cast<double>(cp.size());

      Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
      for (const auto & p : cp) {
        const Eigen::Vector2d d = p - mean;
        cov += d * d.transpose();
      }
      cov /= static_cast<double>(cp.size() - 1);

      // Regularize: floor each eigenvalue to an absolute minimum variance so a
      // thin wall segment keeps a usable (non-degenerate) Gaussian.
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
      Eigen::Vector2d ev = es.eigenvalues();
      if (ev.maxCoeff() <= 0.0) continue;
      for (int k = 0; k < 2; ++k) ev[k] = std::max(ev[k], min_var);
      Eigen::Matrix2d d = Eigen::Matrix2d::Zero();
      d(0, 0) = ev[0];
      d(1, 1) = ev[1];
      const Eigen::Matrix2d cov_reg =
        es.eigenvectors() * d * es.eigenvectors().transpose();

      NdtCell ndt;
      ndt.mean = mean;
      ndt.inv_cov = cov_reg.inverse();
      ndt.valid = true;
      cells_[kv.first] = ndt;
    }
  }

  // Cell containing p, or nullptr if none / not enough support.
  const NdtCell * at(const Eigen::Vector2d & p) const
  {
    auto it = cells_.find(cellOf(p, inv_));
    if (it == cells_.end() || !it->second.valid) return nullptr;
    return &it->second;
  }

  bool empty() const { return cells_.empty(); }
  std::size_t size() const { return cells_.size(); }

private:
  double cell_ = 1.0;
  double inv_ = 1.0;
  std::unordered_map<CellKey, NdtCell, CellKeyHash> cells_;
};

}  // namespace scan_matching_localization
