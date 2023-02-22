#include <cmath>

#include "SimCalorimetry/EcalSimAlgos/interface/ComponentShape.h"

void ComponentShape::fillShape(float& time_interval,
                               double& m_thresh,
                               EcalShapeBase::DVec& aVec,
                               const edm::EventSetup* es) const {
  if (m_useDBShape) {
    if (es == nullptr) {
      throw cms::Exception("[ComponentShape] DB conditions are not available, const edm::EventSetup* es == nullptr ");
    }
    auto const& esps = es->getData(espsToken_);

    //barrel_shapes elements are vectors of floats, to save space in db
    aVec = std::vector<double>(esps.barrel_shapes.at(shapeIndex_).begin(), esps.barrel_shapes.at(shapeIndex_).end());
    time_interval = esps.time_interval;
    m_thresh = esps.barrel_thresh;
  }

  else {  // fill with dummy values, since this code is only reached before the actual digi simulation
    m_thresh = 0.00013;
    time_interval = 1.0;
    aVec.reserve(500);
    for (unsigned int i(0); i != 500; ++i)
      aVec.push_back(0.0);
  }
}

double ComponentShape::timeToRise() const {
  return kTimeToRise;
}  // hardcoded rather than computed because
   // components need relative time shifts
