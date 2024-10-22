#ifndef SimTracker_TrackAssociation_TrackingParticleIP
#define SimTracker_TrackAssociation_TrackingParticleIP

// This file is in this package only because ParametersDefinerForTP is

#include <cmath>

namespace TrackingParticleIP {
  // As in TrackBase::dxy(Point) and dz(Point)
  template <typename T_Vertex, typename T_Momentum, typename T_Point>
  inline auto dxy(const T_Vertex &vertex, const T_Momentum &momentum, const T_Point &point) {
    return -(vertex.x() - point.x()) * std::sin(momentum.phi()) + (vertex.y() - point.y()) * std::cos(momentum.phi());
  }

  template <typename T_Vertex, typename T_Momentum, typename T_Point>
  inline auto dz(const T_Vertex &vertex, const T_Momentum &momentum, const T_Point &point) {
    return vertex.z() - point.z() -
           ((vertex.x() - point.x()) * momentum.x() + (vertex.y() - point.y()) * momentum.y()) * momentum.z() /
               momentum.perp2();
  }
}  // namespace TrackingParticleIP

#endif
