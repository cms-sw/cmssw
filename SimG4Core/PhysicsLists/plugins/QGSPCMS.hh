#ifndef SimG4Core_PhysicsLists_QGSPCMS_H
#define SimG4Core_PhysicsLists_QGSPCMS_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class QGSPCMS: public PhysicsList {

public:
  QGSPCMS(G4LogicalVolumeToDDLogicalPartMap& map, const HepPDT::ParticleDataTable * table_, sim::FieldBuilder *fieldBuilder_, const edm::ParameterSet & p);
};

#endif

