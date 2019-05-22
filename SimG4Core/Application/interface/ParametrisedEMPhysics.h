
#ifndef SimG4Core_GFlash_ParametrisedPhysics_H
#define SimG4Core_GFlash_ParametrisedPhysics_H

// Joanna Weng 08.2005
// Physics process for Gflash parameterisation
// modified by Soon Yung Jun, Dongwook Jang
// V.Ivanchenko rename the class, cleanup, and move
//              to SimG4Core/Application - 2012/08/14

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "G4VPhysicsConstructor.hh"

class G4ParticleDefinition;

class ParametrisedEMPhysics : public G4VPhysicsConstructor {
public:
  ParametrisedEMPhysics(const std::string& name, const edm::ParameterSet& p);
  ~ParametrisedEMPhysics() override;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:
  void ModifyTransportation(const G4ParticleDefinition*, int ntry, double th1, double th2);

  edm::ParameterSet theParSet;

  struct TLSmod;
  static G4ThreadLocal TLSmod* m_tpmod;
};

#endif
