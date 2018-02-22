#ifndef SimG4Core_PhysicsLists_EMPhysics1CMS_H
#define SimG4Core_PhysicsLists_EMPhysics1CMS_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EMPhysics1CMS: public PhysicsList {

public:
  EMPhysics1CMS(G4LogicalVolumeToDDLogicalPartMap& map, const HepPDT::ParticleDataTable * table_, sim::ChordFinderSetter *chordFinderSetter_, const edm::ParameterSet & p);
};

#endif



