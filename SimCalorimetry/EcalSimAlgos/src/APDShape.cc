#include <cmath>

#include "SimCalorimetry/EcalSimAlgos/interface/APDShape.h"

#include <cassert>

void APDShape::fillShape(float& time_interval,
                         double& m_thresh,
                         EcalShapeBase::DVec& aVec,
                         const edm::EventSetup* es) const {
  if (m_useDBShape) {
    if (es == nullptr) {
      throw cms::Exception("EcalShapeBase:: DB conditions are not available, const edm::EventSetup* es == nullptr ");
    }
    edm::ESHandle<EcalSimPulseShape> esps;
    es->get<EcalSimPulseShapeRcd>().get(esps);

    aVec = esps->apd_shape;
    time_interval = esps->time_interval;
    m_thresh = esps->apd_thresh;
  } else {
    m_thresh = 0.0;
    time_interval = 1.0;
    aVec.reserve(500);
    const double m_tStart = 74.5;
    const double m_tau = 40.5;

    for (unsigned int i(0); i != 500; ++i) {
      const double ctime((1. * i + 0.5 - m_tStart) / m_tau);
      double val = 0 > ctime ? 0 : ctime * exp(1. - ctime);
      aVec.push_back(val);
    }
  }
}
