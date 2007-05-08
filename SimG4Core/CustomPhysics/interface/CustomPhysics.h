#ifndef SimG4Core_CustomPhysics_H
#define SimG4Core_CustomPhysics_H
 
#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class CustomPhysics : public PhysicsList
{
public:
    CustomPhysics(G4LogicalVolumeToDDLogicalPartMap& map,
		  const edm::ParameterSet & p);
};
 
#endif
