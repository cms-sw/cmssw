#ifndef SimPPS_PPSPixelDigiProducer_RPix_HIT_CHARGE_CONVERTER_H
#define SimPPS_PPSPixelDigiProducer_RPix_HIT_CHARGE_CONVERTER_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimPPS/PPSPixelDigiProducer/interface/RPixLinearChargeCollectionDrifter.h"
#include "SimPPS/PPSPixelDigiProducer/interface/RPixLinearChargeDivider.h"
#include "SimPPS/PPSPixelDigiProducer/interface/RPixChargeShare.h"
#include "CondFormats/PPSObjects/interface/PPSPixelTopology.h"

class RPixHitChargeConverter {
public:
  RPixHitChargeConverter(const edm::ParameterSet &params_,
                         CLHEP::HepRandomEngine &eng,
                         uint32_t det_id,
                         const PPSPixelTopology &ppt);
  ~RPixHitChargeConverter() = default;

  std::map<unsigned short, double> processHit(const PSimHit &hit, const PPSPixelTopology &ppt);

private:
  const uint32_t det_id_;
  std::unique_ptr<RPixLinearChargeDivider> theRPixChargeDivider;
  std::unique_ptr<RPixLinearChargeCollectionDrifter> theRPixChargeCollectionDrifter;
  std::unique_ptr<RPixChargeShare> theRPixChargeShare;
  int verbosity_;
};

#endif
