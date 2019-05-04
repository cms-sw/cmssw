#ifndef SimG4Core_GFlash_ParametrisedPhysics_H
#define SimG4Core_GFlash_ParametrisedPhysics_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "G4FastSimulationManagerProcess.hh"
#include "G4VPhysicsConstructor.hh"
#include "SimG4Core/GFlash/interface/GflashEMShowerModel.h"
#include "SimG4Core/GFlash/interface/GflashHadronShowerModel.h"

// Joanna Weng 08.2005
// Physics process for Gflash parameterisation
// modified by Soon Yung Jun, Dongwook Jang

class ParametrisedPhysics : public G4VPhysicsConstructor {
public:
  ParametrisedPhysics(std::string name, const edm::ParameterSet &p);
  ~ParametrisedPhysics() override;

protected:
  void ConstructParticle() override;
  void ConstructProcess() override;

private:
  edm::ParameterSet theParSet;
  struct ThreadPrivate {
    GflashEMShowerModel *theEMShowerModel;
    GflashEMShowerModel *theHadShowerModel;
    GflashHadronShowerModel *theHadronShowerModel;
    G4FastSimulationManagerProcess *theFastSimulationManagerProcess;
  };
  static G4ThreadLocal ThreadPrivate *tpdata;
};

#endif
