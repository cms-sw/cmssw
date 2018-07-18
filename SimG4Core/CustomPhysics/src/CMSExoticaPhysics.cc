#include "SimG4Core/CustomPhysics/interface/CMSExoticaPhysics.h"
#include "SimG4Core/CustomPhysics/interface/CustomPhysicsList.h"
#include "SimG4Core/CustomPhysics/interface/CustomPhysicsListSS.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
 
CMSExoticaPhysics::CMSExoticaPhysics(PhysicsList* phys, const edm::ParameterSet & p) {

  bool ssPhys  = p.getUntrackedParameter<bool>("ExoticaPhysicsSS",false);
                 
  if(ssPhys) {
    phys->RegisterPhysics(new CustomPhysicsListSS("custom",p,true));
  } else {
    phys->RegisterPhysics(new CustomPhysicsList("custom",p,true));    
  }
}
