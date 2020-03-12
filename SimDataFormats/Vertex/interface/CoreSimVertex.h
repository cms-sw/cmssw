#ifndef CoreSimVertex_H
#define CoreSimVertex_H

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include <cmath>

/**  a generic Simulated Vertex
 */
class CoreSimVertex {
public:
  /// constructors
  CoreSimVertex() {}

  CoreSimVertex(const math::XYZVectorD& v, float tof) { theVertex.SetXYZT(v.x(), v.y(), v.z(), tof); }

  CoreSimVertex(const math::XYZTLorentzVectorD& v) { theVertex.SetXYZT(v.x(), v.y(), v.z(), v.t()); }

  const math::XYZTLorentzVectorD& position() const { return theVertex; }

  void setEventId(EncodedEventId e) { eId = e; }

  EncodedEventId eventId() const { return eId; }

  void setTof(float tof) { theVertex.SetXYZT(theVertex.x(), theVertex.y(), theVertex.z(), tof); }

private:
  EncodedEventId eId;
  math::XYZTLorentzVectorD theVertex;
};

#include <iosfwd>
std::ostream& operator<<(std::ostream& o, const CoreSimVertex& v);

#endif
