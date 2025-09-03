// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/scene/projection.h"

#include <limits>

namespace colmap {

double CalculateSquaredReprojectionError(const Eigen::Vector2d& point2D,
                                         const Eigen::Vector3d& point3D,
                                         const Rigid3d& cam_from_world,
                                         const Camera& camera) {
  // For spherical cameras, use angular error converted to pixel scale
  const double angular_error = CalculateAngularReprojectionError(
      point2D, point3D, cam_from_world, camera);
  if (angular_error == EIGEN_PI) {
    return std::numeric_limits<double>::max();
  }
  
  // Convert angular error to pixel scale
  // Use image resolution to scale angular error to pixel units
  const double width = camera.width;
  const double height = camera.height;
  const double scale_factor = std::max(width, height) / EIGEN_PI;
  const double pixel_error = angular_error * scale_factor;
  
  return pixel_error * pixel_error;
}

double CalculateSquaredReprojectionError(
    const Eigen::Vector2d& point2D,
    const Eigen::Vector3d& point3D,
    const Eigen::Matrix3x4d& cam_from_world,
    const Camera& camera) {
  // For spherical cameras, use angular error converted to pixel scale
  const double angular_error = CalculateAngularReprojectionError(
      point2D, point3D, cam_from_world, camera);
  if (angular_error == EIGEN_PI) {
    return std::numeric_limits<double>::max();
  }
  
  // Convert angular error to pixel scale
  // Use image resolution to scale angular error to pixel units
  const double width = camera.width;
  const double height = camera.height;
  const double scale_factor = std::max(width, height) / EIGEN_PI;
  const double pixel_error = angular_error * scale_factor;
  
  return pixel_error * pixel_error;
}

double CalculateAngularReprojectionError(const Eigen::Vector2d& point2D,
                                         const Eigen::Vector3d& point3D,
                                         const Rigid3d& cam_from_world,
                                         const Camera& camera) {
  // For spherical cameras, CamFromImg now returns 3D ray direction
  const std::optional<Eigen::Vector3d> cam_ray = camera.CamFromImg(point2D);
  if (!cam_ray) {
    return EIGEN_PI;
  }
  // Ensure the ray is normalized
  const Eigen::Vector3d normalized_cam_ray = cam_ray->normalized();
  return CalculateAngularReprojectionError(
      normalized_cam_ray, point3D, cam_from_world);
}

double CalculateAngularReprojectionError(
    const Eigen::Vector2d& point2D,
    const Eigen::Vector3d& point3D,
    const Eigen::Matrix3x4d& cam_from_world,
    const Camera& camera) {
  // For spherical cameras, CamFromImg now returns 3D ray direction
  const std::optional<Eigen::Vector3d> cam_ray = camera.CamFromImg(point2D);
  if (!cam_ray) {
    return EIGEN_PI;
  }
  // Ensure the ray is normalized
  const Eigen::Vector3d normalized_cam_ray = cam_ray->normalized();
  return CalculateAngularReprojectionError(
      normalized_cam_ray, point3D, cam_from_world);
}

double CalculateAngularReprojectionError(const Eigen::Vector3d& cam_ray,
                                         const Eigen::Vector3d& point3D,
                                         const Rigid3d& cam_from_world) {
  const Eigen::Vector3d point3D_in_cam = cam_from_world * point3D;
  const double dot_product = std::max(-1.0, std::min(1.0, 
      (cam_ray.transpose() * point3D_in_cam.normalized()).value()));
  return std::acos(dot_product);
}

double CalculateAngularReprojectionError(
    const Eigen::Vector3d& cam_ray,
    const Eigen::Vector3d& point3D,
    const Eigen::Matrix3x4d& cam_from_world) {
  // Transform point3D to homogeneous coordinates and apply transformation
  const Eigen::Vector4d point3D_homo = point3D.homogeneous();
  const Eigen::Vector3d point3D_in_cam = cam_from_world * point3D_homo;
  const double dot_product = std::max(-1.0, std::min(1.0,
      (cam_ray.transpose() * point3D_in_cam.normalized()).value()));
  return std::acos(dot_product);
}
bool HasPointPositiveDepth(const Eigen::Matrix3x4d& cam_from_world,
                           const Eigen::Vector3d& point3D) {
  // For spherical cameras, all points are "visible" since it's omnidirectional
  // But we still want to check if the point is at a reasonable distance
  const Eigen::Vector3d point3D_in_cam = 
      cam_from_world * point3D.homogeneous();
  
  // Check if point is not at the camera center (which would be undefined)
  return point3D_in_cam.norm() > std::numeric_limits<double>::epsilon();
}

}  // namespace colmap