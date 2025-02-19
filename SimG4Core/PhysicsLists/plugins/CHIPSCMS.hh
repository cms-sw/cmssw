#ifndef SimG4Core_PhysicsLists_CHIPSCMS_H
#define SimG4Core_PhysicsLists_CHIPSCMS_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CHIPSCMS: public PhysicsList {

public:
  CHIPSCMS(G4LogicalVolumeToDDLogicalPartMap& map, const HepPDT::ParticleDataTable * table_, sim::FieldBuilder *fieldBuilder_, const edm::ParameterSet & p);
};

#endif

