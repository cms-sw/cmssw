#ifndef SimG4Core_PhysicsLists_CMSModel_H
#define SimG4Core_PhysicsLists_CMSModel_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CMSModel: public PhysicsList {

public:
  CMSModel(G4LogicalVolumeToDDLogicalPartMap& map, const HepPDT::ParticleDataTable * table_, sim::FieldBuilder *fieldBuilder_, const edm::ParameterSet & p);
};

#endif

