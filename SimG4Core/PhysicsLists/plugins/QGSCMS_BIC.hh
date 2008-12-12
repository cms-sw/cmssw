#ifndef SimG4Core_PhysicsLists_QGSCMS_BIC_H
#define SimG4Core_PhysicsLists_QGSCMS_BIC_H
 
#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class QGSCMS_BIC: public PhysicsList {

public:
  QGSCMS_BIC(G4LogicalVolumeToDDLogicalPartMap& map, 
	     const edm::ParameterSet & p);
};

#endif
