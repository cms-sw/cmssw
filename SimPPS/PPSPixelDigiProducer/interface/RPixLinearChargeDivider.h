#ifndef SimPPS_PPSPixelDigiProducer_LINEAR_CHARGE_DIVIDER_H
#define SimPPS_PPSPixelDigiProducer_LINEAR_CHARGE_DIVIDER_H

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/Common/interface/SiG4UniversalFluctuation.h"
#include "SimPPS/PPSPixelDigiProducer/interface/RPixEnergyDepositUnit.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace CLHEP {
  class HepRandomEngine;
}

class RPixLinearChargeDivider {
public:
  RPixLinearChargeDivider(const edm::ParameterSet& params, CLHEP::HepRandomEngine& eng, uint32_t det_id);
  ~RPixLinearChargeDivider();

  std::vector<RPixEnergyDepositUnit> divide(const PSimHit& hit);

private:
  CLHEP::HepRandomEngine& rndEngine_;
  uint32_t det_id_;
  bool fluctuateCharge_;
  int chargedivisions_;
  double deltaCut_;
  double pitch_;
  double thickness_;

  std::vector<RPixEnergyDepositUnit> the_energy_path_distribution_;
  SiG4UniversalFluctuation* fluctuate;
  int verbosity_;

  void FluctuateEloss(int pid,
                      double particleMomentum,
                      double eloss,
                      double length,
                      int NumberOfSegs,
                      std::vector<RPixEnergyDepositUnit>& elossVector);
};

#endif
