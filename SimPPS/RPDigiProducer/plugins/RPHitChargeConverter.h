#ifndef SimPPS_RPDigiProducer_RP_HIT_CHARGE_CONVERTER_H
#define SimPPS_RPDigiProducer_RP_HIT_CHARGE_CONVERTER_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimPPS/RPDigiProducer/plugins/RPLinearChargeCollectionDrifter.h"
#include "SimPPS/RPDigiProducer/plugins/RPLinearChargeDivider.h"
#include "SimPPS/RPDigiProducer/plugins/RPLinearInduceChargeOnStrips.h"
#include "SimPPS/RPDigiProducer/interface/RPSimTypes.h"

#include <map>

class RPHitChargeConverter {
public:
  RPHitChargeConverter(const edm::ParameterSet &params_, CLHEP::HepRandomEngine &eng, RPDetId det_id);
  ~RPHitChargeConverter();

  simromanpot::strip_charge_map processHit(const PSimHit &hit);

private:
  const RPDetId det_id_;

  std::unique_ptr<RPLinearChargeDivider> theRPChargeDivider;
  std::unique_ptr<RPLinearChargeCollectionDrifter> theRPChargeCollectionDrifter;
  std::unique_ptr<RPLinearInduceChargeOnStrips> theRPInduceChargeOnStrips;
  int verbosity_;
};

#endif  //SimPPS_RPDigiProducer_RP_HIT_CHARGE_CONVERTER_H
