#ifndef SimG4Core_PhysicsLists_QGSPCMS_G4v52_H
#define SimG4Core_PhysicsLists_QGSPCMS_G4v52_H
 
#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class QGSPCMS_G4v52: public PhysicsList {

public:
  QGSPCMS_G4v52(G4LogicalVolumeToDDLogicalPartMap& map, 
		const edm::ParameterSet & p);
};
 
#endif


