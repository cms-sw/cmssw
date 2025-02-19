#ifndef SimG4Core_PhysicsLists_QGSPCMS_BERT_EML_H
#define SimG4Core_PhysicsLists_QGSPCMS_BERT_EML_H
 
#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class QGSPCMS_BERT_EML: public PhysicsList {

public:
  QGSPCMS_BERT_EML(G4LogicalVolumeToDDLogicalPartMap& map, const HepPDT::ParticleDataTable * table_, sim::FieldBuilder *fieldBuilder_, const edm::ParameterSet & p);
};

#endif


