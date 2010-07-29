#ifndef SimG4Core_PhysicsLists_QBBCCMS_H
#define SimG4Core_PhysicsLists_QBBCCMS_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class QBBCCMS: public PhysicsList {

public:
  QBBCCMS(G4LogicalVolumeToDDLogicalPartMap& map, const HepPDT::ParticleDataTable * table_, sim::FieldBuilder *fieldBuilder_, const edm::ParameterSet & p);
};

#endif

