#ifndef SimPPS_RPDigiProducer_RP_LINEAR_CHARGE_DIVIDER_H
#define SimPPS_RPDigiProducer_RP_LINEAR_CHARGE_DIVIDER_H

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/Common/interface/SiG4UniversalFluctuation.h"
#include "SimPPS/RPDigiProducer/interface/RPSimTypes.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace CLHEP{
	class HepRandomEngine;
}


class RPLinearChargeDivider
{
  public:
    RPLinearChargeDivider(const edm::ParameterSet &params, CLHEP::HepRandomEngine& eng, RPDetId det_id);
    ~RPLinearChargeDivider();
    SimRP::energy_path_distribution divide(const PSimHit& hit);
  private:
    const edm::ParameterSet &params_;
    CLHEP::HepRandomEngine& rndEngine;
    RPDetId _det_id;

    bool fluctuateCharge_;
    int chargedivisionsPerStrip_;
    int chargedivisionsPerThickness_;
    double deltaCut_;
    double pitch_;
    double thickness_;
    SimRP::energy_path_distribution the_energy_path_distribution_;
    SiG4UniversalFluctuation * fluctuate; 
    int verbosity_;
    
    void FluctuateEloss(int pid, double particleMomentum, 
        double eloss, double length, int NumberOfSegs, 
        SimRP::energy_path_distribution &elossVector);
};

#endif  //SimPPS_RPDigiProducer_RP_LINEAR_CHARGE_DIVIDER_H
