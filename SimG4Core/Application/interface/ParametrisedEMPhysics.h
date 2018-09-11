#ifndef SimG4Core_GFlash_ParametrisedPhysics_H
#define SimG4Core_GFlash_ParametrisedPhysics_H

// Joanna Weng 08.2005
// Physics process for Gflash parameterisation
// modified by Soon Yung Jun, Dongwook Jang
// V.Ivanchenko rename the class, cleanup, and move
//              to SimG4Core/Application - 2012/08/14

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4VPhysicsConstructor.hh"

class GFlashEMShowerModel;
class GFlashHadronShowerModel;
class ElectronLimiter;
class G4FastSimulationManagerProcess;

class ParametrisedEMPhysics : public G4VPhysicsConstructor
{
public:

  ParametrisedEMPhysics(std::string name, const edm::ParameterSet & p);
  ~ParametrisedEMPhysics() override;
	
  void ConstructParticle() override;
  void ConstructProcess() override;

private:

  edm::ParameterSet theParSet;

  static G4ThreadLocal std::unique_ptr<GFlashEMShowerModel> theEcalEMShowerModel;
  static G4ThreadLocal std::unique_ptr<GFlashEMShowerModel> theHcalEMShowerModel;
  static G4ThreadLocal std::unique_ptr<GFlashHadronShowerModel> theEcalHadShowerModel;
  static G4ThreadLocal std::unique_ptr<GFlashHadronShowerModel> theHcalHadShowerModel;

  static G4ThreadLocal std::unique_ptr<ElectronLimiter> theElectronLimiter;
  static G4ThreadLocal std::unique_ptr<ElectronLimiter> thePositronLimiter;

  static G4ThreadLocal std::unique_ptr<G4FastSimulationManagerProcess> theFastSimulationManagerProcess; 

};

#endif

