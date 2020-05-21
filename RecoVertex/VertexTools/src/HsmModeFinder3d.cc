#include "RecoVertex/VertexTools/interface/hsm_3d.h"
#include "RecoVertex/VertexTools/interface/HsmModeFinder3d.h"

/** Half sample mode in 3d, as a functional class.
 */

GlobalPoint HsmModeFinder3d::operator()(const std::vector<PointAndDistance>& values) const {
  std::vector<GlobalPoint> v;
  for (const auto& value : values) {
    v.push_back(value.first);
  };
  return hsm_3d(v);
}

HsmModeFinder3d* HsmModeFinder3d::clone() const { return new HsmModeFinder3d(*this); }
