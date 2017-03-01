#ifndef SimG4Core_PhysicsLists_QGSPCMS_BERT_EML95_H
#define SimG4Core_PhysicsLists_QGSPCMS_BERT_EML95_H
 
#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class QGSPCMS_BERT_EML95: public PhysicsList {

public:
  QGSPCMS_BERT_EML95(G4LogicalVolumeToDDLogicalPartMap& map, const HepPDT::ParticleDataTable * table_, sim::ChordFinderSetter *chordFinderSetter_, const edm::ParameterSet & p);
};

#endif


