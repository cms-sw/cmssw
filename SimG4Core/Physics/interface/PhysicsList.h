#ifndef SimG4Core_PhysicsList_H
#define SimG4Core_PhysicsList_H

#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4VModularPhysicsList.hh"

class DDG4ProductionCuts;

class PhysicsList : public G4VModularPhysicsList
{
public:
  PhysicsList(G4LogicalVolumeToDDLogicalPartMap & map,
	      const edm::ParameterSet & p);
  virtual ~PhysicsList();
  virtual void SetCuts();
private:
  G4LogicalVolumeToDDLogicalPartMap map_;
  edm::ParameterSet m_pPhysics; 
  DDG4ProductionCuts * prodCuts;
};

#endif
