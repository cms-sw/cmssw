#ifndef SimPPS_PPSPixelDigiProducer_RPix_LINEAR_CHARGE_COLLECTION_DRIFTER_H
#define SimPPS_PPSPixelDigiProducer_RPix_LINEAR_CHARGE_COLLECTION_DRIFTER_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimPPS/PPSPixelDigiProducer/interface/RPixSignalPoint.h"
#include "SimPPS/PPSPixelDigiProducer/interface/RPixEnergyDepositUnit.h"
#include "CondFormats/PPSObjects/interface/PPSPixelTopology.h"

class RPixLinearChargeCollectionDrifter {
public:
  RPixLinearChargeCollectionDrifter(const edm::ParameterSet &params, uint32_t det_id, const PPSPixelTopology &ppt);
  std::vector<RPixSignalPoint> Drift(const std::vector<RPixEnergyDepositUnit> &energy_deposition);

private:
  std::vector<RPixSignalPoint> temp_;

  std::vector<double> charge_cloud_sigmas_vect_;
  double GeV_per_electron_;
  int verbosity_;
  double det_thickness_;
  uint32_t det_id_;

  double getSigma_(double z);
};

#endif
