#ifndef SimG4Core_PhysicsLists_QGSPCMS_EMV_H
#define SimG4Core_PhysicsLists_QGSPCMS_EMV_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class QGSPCMS_EMV: public PhysicsList {

public:
  QGSPCMS_EMV(G4LogicalVolumeToDDLogicalPartMap& map, 
	      const edm::ParameterSet & p);
};

#endif

