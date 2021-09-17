#ifndef SimPPS_RPDigiProducer_RP_LINEAR_CHARGE_COLLECTION_DRIFTER_H
#define SimPPS_RPDigiProducer_RP_LINEAR_CHARGE_COLLECTION_DRIFTER_H

#include <vector>
#include <iostream>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimPPS/RPDigiProducer/interface/RPSimTypes.h"

class RPLinearChargeCollectionDrifter {
public:
  RPLinearChargeCollectionDrifter(const edm::ParameterSet &params, RPDetId det_id);
  simromanpot::charge_induced_on_surface Drift(const simromanpot::energy_path_distribution &energy_deposition);

private:
  std::vector<double> charge_cloud_sigmas_vect_;
  double GeV_per_electron_;
  int verbosity_;
  double det_thickness_;
  RPDetId det_id_;

  double getSigma(double z);  //z - z position
};

#endif  //SimPPS_RPDigiProducer_RP_LINEAR_CHARGE_COLLECTION_DRIFTER_H
