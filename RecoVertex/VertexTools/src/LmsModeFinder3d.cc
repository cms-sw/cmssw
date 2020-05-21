#include "RecoVertex/VertexTools/interface/lms_3d.h"
#include "RecoVertex/VertexTools/interface/LmsModeFinder3d.h"

GlobalPoint LmsModeFinder3d::operator()(const std::vector<PointAndDistance>& values) const {
  std::vector<GlobalPoint> v;
  for (const auto& value : values) {
    v.push_back(value.first);
  };
  return lms_3d(v);
}

LmsModeFinder3d* LmsModeFinder3d::clone() const { return new LmsModeFinder3d(*this); }
