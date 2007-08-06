#ifndef SimG4Core_PhysicsLists_LHEPCMS_H
#define SimG4Core_PhysicsLists_LHEPCMS_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class LHEPCMS: public PhysicsList {

public:
  LHEPCMS(G4LogicalVolumeToDDLogicalPartMap& map, const edm::ParameterSet & p);
};

#endif



