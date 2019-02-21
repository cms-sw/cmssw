#ifndef SimPPS_RPDigiProducer_RP_HIT_CHARGE_CONVERTER_H
#define SimPPS_RPDigiProducer_RP_HIT_CHARGE_CONVERTER_H

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include <map>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimPPS/RPDigiProducer/interface/RPLinearChargeCollectionDrifter.h"
#include "SimPPS/RPDigiProducer/interface/RPLinearChargeDivider.h"
#include "SimPPS/RPDigiProducer/interface/RPLinearInduceChargeOnStrips.h"
#include "SimPPS/RPDigiProducer/interface/RPSimTypes.h"

class RPHitChargeConverter
{
  public:
    RPHitChargeConverter(const edm::ParameterSet &params_, CLHEP::HepRandomEngine& eng, RPDetId det_id);
    ~RPHitChargeConverter();
    
    SimRP::strip_charge_map processHit(const PSimHit &hit);
  private:
    const edm::ParameterSet &params_;
    const RPDetId det_id_;
    
    RPLinearChargeDivider* theRPChargeDivider;
    RPLinearChargeCollectionDrifter* theRPChargeCollectionDrifter;
    RPLinearInduceChargeOnStrips* theRPInduceChargeOnStrips;
    int verbosity_;
};

#endif  //SimPPS_RPDigiProducer_RP_HIT_CHARGE_CONVERTER_H
