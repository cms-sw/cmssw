#include <cmath>

#include "SimCalorimetry/EcalSimAlgos/interface/ComponentShape.h"

// #define component_shape_debug 1

void ComponentShape::fillShape(float& time_interval,
                               double& m_thresh,
                               EcalShapeBase::DVec& aVec,
                               const edm::EventSetup* es) const {
#ifdef component_shape_debug
  std::cout << "ComponentShape::fillShape called with m_useDBShape = " << m_useDBShape << " m_thresh = " << m_thresh
            << " time_interval = " << time_interval << std::endl;
#endif
  if (m_useDBShape) {
#ifdef component_shape_debug
    std::cout << ("[ComponentShape] about to check if es == nullptr") << std::endl;
#endif

    if (es == nullptr) {
      throw cms::Exception("[ComponentShape] DB conditions are not available, const edm::EventSetup* es == nullptr ");
    }
#ifdef component_shape_debug
    std::cout << "ComponentShape::fillShape about to call es->getData(espsToken_)" << m_useDBShape << std::endl;
#endif
    auto const& esps = es->getData(espsToken_);

    aVec = esps.barrel_shapes.at(shapeIndex_);
    time_interval = esps.time_interval;
    m_thresh = esps.barrel_thresh;

#ifdef component_shape_debug
    std::cout << " time_interval = " << time_interval << " m_thresh = " << m_thresh << std::endl;
#endif

  }

  else {  // fill with dummy values, since this code is only reached before the actual digi simulation

#ifdef component_shape_debug
    std::cout << ("[ComponentShape] calling fillShape with m_useDBShape==false, this should do nothing?");
#endif

    m_thresh = 0.00013;
    time_interval = 1.0;
    aVec.reserve(500);
    for (unsigned int i(0); i != 500; ++i)
      aVec.push_back(0.0);
  }
}

double ComponentShape::timeToRise() const {
  return 16.0;
}  // hardcoded rather than computed because
   // components need relative time shifts
   // 16 nanoseconds ~aligns the phase II component
   // sim to the default with the current setup
