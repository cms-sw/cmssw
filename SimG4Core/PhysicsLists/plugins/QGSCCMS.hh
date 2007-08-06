#ifndef SimG4Core_PhysicsLists_QGSCCMS_H
#define SimG4Core_PhysicsLists_QGSCCMS_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class QGSCCMS: public PhysicsList {

public:
  QGSCCMS(G4LogicalVolumeToDDLogicalPartMap& map, const edm::ParameterSet & p);
};

#endif

