#ifndef SimG4Core_PhysicsLists_EMPhysics2CMS_H
#define SimG4Core_PhysicsLists_EMPhysics2CMS_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EMPhysics2CMS: public PhysicsList {

public:
  EMPhysics2CMS(G4LogicalVolumeToDDLogicalPartMap& map, const HepPDT::ParticleDataTable * table_, sim::ChordFinderSetter *chordFinderSetter_, const edm::ParameterSet & p);
};

#endif



