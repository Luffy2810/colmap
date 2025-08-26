// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// ... (license header)

#include "colmap/geometry/triangulation.h"

#include "colmap/geometry/pose.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Dense>

namespace colmap {

bool TriangulatePoint(const Eigen::Matrix3x4d& cam1_from_world,
                      const Eigen::Matrix3x4d& cam2_from_world,
                      const Eigen::Vector3d& cam_ray1,
                      const Eigen::Vector3d& cam_ray2,
                      Eigen::Vector3d* xyz) {
  THROW_CHECK_NOTNULL(xyz);

  Eigen::Matrix4d A;

  A.row(0) = cam_ray1.y() * cam1_from_world.row(0) - cam_ray1.x() * cam1_from_world.row(1);
  A.row(1) = cam_ray1.z() * cam1_from_world.row(0) - cam_ray1.x() * cam1_from_world.row(2);
  A.row(2) = cam_ray2.y() * cam2_from_world.row(0) - cam_ray2.x() * cam2_from_world.row(1);
  A.row(3) = cam_ray2.z() * cam2_from_world.row(0) - cam_ray2.x() * cam2_from_world.row(2);

  const Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
  if (svd.info() != Eigen::Success) {
    return false;
  }
#endif

  const Eigen::Vector4d world_point_homogeneous = svd.matrixV().col(3);

  if (world_point_homogeneous(3) == 0) {
    return false;
  }

  *xyz = world_point_homogeneous.hnormalized();
  return true;
}

bool TriangulateMidPoint(const Rigid3d& cam2_from_cam1,
                         const Eigen::Vector3d& cam_ray1,
                         const Eigen::Vector3d& cam_ray2,
                         Eigen::Vector3d* point3D_in_cam1) {
  const Eigen::Vector3d t12 = cam2_from_cam1.translation;
  const Eigen::Vector3d d1 = cam_ray1.normalized();
  const Eigen::Vector3d d2 = (cam2_from_cam1.rotation * cam_ray2).normalized();

  const Eigen::Vector3d d1_cross_d2 = d1.cross(d2);
  const double d1_cross_d2_norm_sq = d1_cross_d2.squaredNorm();

  if (d1_cross_d2_norm_sq < std::numeric_limits<double>::epsilon()) {
      return false;
  }

  const double t1 = (t12.cross(d2)).dot(d1_cross_d2) / d1_cross_d2_norm_sq;
  const double t2 = (t12.cross(d1)).dot(d1_cross_d2) / d1_cross_d2_norm_sq;

  if (t1 <= std::numeric_limits<double>::epsilon() || t2 <= std::numeric_limits<double>::epsilon()) {
    return false;
  }

  const Eigen::Vector3d p1 = t1 * d1;
  const Eigen::Vector3d p2 = t12 + t2 * d2;

  *point3D_in_cam1 = 0.5 * (p1 + p2);

  return true;
}

bool TriangulateMultiViewPoint(
    const span<const Eigen::Matrix3x4d>& cams_from_world,
    const span<const Eigen::Vector3d>& cam_rays,
    Eigen::Vector3d* xyz) {
  THROW_CHECK_EQ(cams_from_world.size(), cam_rays.size());
  THROW_CHECK_GE(cams_from_world.size(), 2);
  THROW_CHECK_NOTNULL(xyz);

  const size_t num_views = cams_from_world.size();
  Eigen::MatrixXd A(2 * num_views, 4);

  for (size_t i = 0; i < num_views; ++i) {
    const Eigen::Vector3d& ray = cam_rays[i];
    const Eigen::Matrix3x4d& cam_from_world = cams_from_world[i];
    A.row(2 * i) = ray.y() * cam_from_world.row(0) - ray.x() * cam_from_world.row(1);
    A.row(2 * i + 1) = ray.z() * cam_from_world.row(0) - ray.x() * cam_from_world.row(2);
  }

  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
  if (svd.info() != Eigen::Success) {
    return false;
  }
#endif
  
  const Eigen::Vector4d world_point_homogeneous = svd.matrixV().col(3);

  if (world_point_homogeneous(3) == 0) {
    return false;
  }

  *xyz = world_point_homogeneous.hnormalized();
  return true;
}

bool TriangulateOptimalPoint(const Eigen::Matrix3x4d& cam1_from_world_mat,
                             const Eigen::Matrix3x4d& cam2_from_world_mat,
                             const Eigen::Vector3d& cam_ray1,
                             const Eigen::Vector3d& cam_ray2,
                             Eigen::Vector3d* xyz) {
  THROW_CHECK_NOTNULL(xyz);

  const Rigid3d cam1_from_world(
      Eigen::Quaterniond(cam1_from_world_mat.leftCols<3>()),
      cam1_from_world_mat.col(3));
  const Rigid3d cam2_from_world(
      Eigen::Quaterniond(cam2_from_world_mat.leftCols<3>()),
      cam2_from_world_mat.col(3));

  const Rigid3d cam2_from_cam1 = cam2_from_world * Inverse(cam1_from_world);
  
  Eigen::Vector3d point3D_in_cam1;
  if (!TriangulateMidPoint(cam2_from_cam1, cam_ray1, cam_ray2, &point3D_in_cam1)) {
    return false;
  }
  
  const Rigid3d world_from_cam1 = Inverse(cam1_from_world);
  *xyz = world_from_cam1 * point3D_in_cam1;
  
  return true;
}


namespace {
inline double CalculateTriangulationAngleWithKnownBaseline(
    double baseline_length_squared,
    const Eigen::Vector3d& proj_center1,
    const Eigen::Vector3d& proj_center2,
    const Eigen::Vector3d& point3D) {
  const double ray_length_squared1 = (point3D - proj_center1).squaredNorm();
  const double ray_length_squared2 = (point3D - proj_center2).squaredNorm();

  const double denominator =
      2.0 * std::sqrt(ray_length_squared1 * ray_length_squared2);
  if (denominator == 0.0) {
    return 0.0;
  }
  const double nominator =
      ray_length_squared1 + ray_length_squared2 - baseline_length_squared;
  const double angle =
      std::acos(std::clamp(nominator / denominator, -1.0, 1.0));

  return std::min(angle, M_PI - angle);
}
}  // namespace

double CalculateTriangulationAngle(const Eigen::Vector3d& proj_center1,
                                   const Eigen::Vector3d& proj_center2,
                                   const Eigen::Vector3d& point3D) {
  const double baseline_length_squared =
      (proj_center1 - proj_center2).squaredNorm();
  return CalculateTriangulationAngleWithKnownBaseline(
      baseline_length_squared, proj_center1, proj_center2, point3D);
}

std::vector<double> CalculateTriangulationAngles(
    const Eigen::Vector3d& proj_center1,
    const Eigen::Vector3d& proj_center2,
    const std::vector<Eigen::Vector3d>& points3D) {
  const double baseline_length_squared =
      (proj_center1 - proj_center2).squaredNorm();
  std::vector<double> angles(points3D.size());
  for (size_t i = 0; i < points3D.size(); ++i) {
    angles[i] = CalculateTriangulationAngleWithKnownBaseline(
        baseline_length_squared, proj_center1, proj_center2, points3D[i]);
  }
  return angles;
}

}  // namespace colmap