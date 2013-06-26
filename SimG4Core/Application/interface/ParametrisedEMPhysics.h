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

class ParametrisedEMPhysics : public G4VPhysicsConstructor
{
public:

  ParametrisedEMPhysics(std::string name, const edm::ParameterSet & p);
  virtual ~ParametrisedEMPhysics();
	
  virtual void ConstructParticle();
  virtual void ConstructProcess();

private:

  edm::ParameterSet theParSet;

  GFlashEMShowerModel *theEMShowerModel;
  GFlashEMShowerModel *theHadShowerModel;

};

#endif

