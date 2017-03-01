#ifndef SimG4Core_PhysicsLists_FTFPCMS_BERT_H
#define SimG4Core_PhysicsLists_FTFPCMS_BERT_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class FTFPCMS_BERT: public PhysicsList {

public:
  FTFPCMS_BERT(G4LogicalVolumeToDDLogicalPartMap& map, const HepPDT::ParticleDataTable * table_, sim::ChordFinderSetter *chordFinderSetter_, const edm::ParameterSet & p);
};

#endif



