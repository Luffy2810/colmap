// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// ... (license header)

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Triangulate a 3D point from two views using the Direct Linear Transform (DLT) method.
bool TriangulatePoint(const Eigen::Matrix3x4d& cam1_from_world,
                      const Eigen::Matrix3x4d& cam2_from_world,
                      const Eigen::Vector3d& cam_ray1,
                      const Eigen::Vector3d& cam_ray2,
                      Eigen::Vector3d* xyz);

// Triangulate the midpoint of the shortest line segment connecting two 3D rays.
bool TriangulateMidPoint(const Rigid3d& cam2_from_cam1,
                         const Eigen::Vector3d& cam_ray1,
                         const Eigen::Vector3d& cam_ray2,
                         Eigen::Vector3d* point3D_in_cam1);

// Triangulate a 3D point from multiple views using the DLT method.
bool TriangulateMultiViewPoint(
    const span<const Eigen::Matrix3x4d>& cams_from_world,
    const span<const Eigen::Vector3d>& cam_rays,
    Eigen::Vector3d* xyz);

// Triangulates an optimal 3D point by finding the midpoint of the shortest
// line segment between two 3D rays in space. This is a robust replacement for
// traditional optimal triangulation methods based on epipolar geometry.
bool TriangulateOptimalPoint(const Eigen::Matrix3x4d& cam1_from_world_mat,
                             const Eigen::Matrix3x4d& cam2_from_world_mat,
                             const Eigen::Vector3d& cam_ray1,
                             const Eigen::Vector3d& cam_ray2,
                             Eigen::Vector3d* xyz);

// Calculate the triangulation angle between two camera rays.
double CalculateTriangulationAngle(const Eigen::Vector3d& proj_center1,
                                   const Eigen::Vector3d& proj_center2,
                                   const Eigen::Vector3d& point3D);

// Vectorized version of CalculateTriangulationAngle.
std::vector<double> CalculateTriangulationAngles(
    const Eigen::Vector3d& proj_center1,
    const Eigen::Vector3d& proj_center2,
    const std::vector<Eigen::Vector3d>& points3D);

}  // namespace colmap