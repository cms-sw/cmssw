#ifndef SimG4Core_PhysicsLists_LHEPCMS_EMV_H
#define SimG4Core_PhysicsLists_LHEPCMS_EMV_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class LHEPCMS_EMV: public PhysicsList {

public:
  LHEPCMS_EMV(G4LogicalVolumeToDDLogicalPartMap& map, const HepPDT::ParticleDataTable * table_, sim::FieldBuilder *fieldBuilder_, const edm::ParameterSet & p);
};

#endif



