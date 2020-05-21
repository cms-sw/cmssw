#include "RecoVertex/VertexTools/interface/SmsModeFinder3d.h"

SmsModeFinder3d::SmsModeFinder3d(const SMS& algo) : theAlgo(algo) {}

GlobalPoint SmsModeFinder3d::operator()(const std::vector<PointAndDistance>& values) const {
  std::vector<std::pair<GlobalPoint, float> > weighted;
  for (const auto& value : values) {
    float weight = pow(10 + 10000 * value.second, -2);
    weighted.push_back(std::pair<GlobalPoint, float>(value.first, weight));
  };
  return theAlgo.location(weighted);
}

SmsModeFinder3d* SmsModeFinder3d::clone() const { return new SmsModeFinder3d(*this); }
