#include "DummyPhysics.h"
#include "SimG4Core/PhysicsLists/interface/DummyEMPhysics.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "G4DecayPhysics.hh"

DummyPhysics::DummyPhysics(const edm::ParameterSet & p) 
  : PhysicsList(p) {

  bool emPhys  = p.getUntrackedParameter<bool>("EMPhysics",true);
  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  if (emPhys) {
    RegisterPhysics(new DummyEMPhysics(ver));
  }
  RegisterPhysics(new G4DecayPhysics(ver));
  edm::LogInfo("PhysicsList") << "DummyPhysics constructed with EM Physics "
			      << emPhys << " and Decay";
}

