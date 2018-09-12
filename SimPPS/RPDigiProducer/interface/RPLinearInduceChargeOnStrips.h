#ifndef SimPPS_RPDigiProducer_RP_LINEAR_INDUCE_CHARGE_ON_STRIPS_H
#define SimPPS_RPDigiProducer_RP_LINEAR_INDUCE_CHARGE_ON_STRIPS_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include "SimPPS/RPDigiProducer/interface/RPSimTypes.h"
#include "Geometry/VeryForwardRPTopology/interface/RPSimTopology.h"

class RPLinearInduceChargeOnStrips
{
  public:
    RPLinearInduceChargeOnStrips(const edm::ParameterSet &params, RPDetId det_id);
    SimRP::strip_charge_map 
        Induce(const SimRP::charge_induced_on_surface &charge_map);
  private:
    RPDetId det_id_;
    std::vector<double> signalCoupling_;
    SimRP::strip_charge_map theStripChargeMap;
    RPSimTopology theRPDetTopology_;
    double sqrt_2;
    int no_of_strips_;
    int verbosity_;
};

#endif
