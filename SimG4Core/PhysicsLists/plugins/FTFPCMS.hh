#ifndef SimG4Core_PhysicsLists_FTFPCMS_H
#define SimG4Core_PhysicsLists_FTFPCMS_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class FTFPCMS: public PhysicsList {

public:
  FTFPCMS(G4LogicalVolumeToDDLogicalPartMap& map, const HepPDT::ParticleDataTable * table_, sim::FieldBuilder *fieldBuilder_, const edm::ParameterSet & p);
};

#endif



