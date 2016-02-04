#ifndef SimG4Core_PhysicsLists_QGSBCMS_BERT_NOLEP1_EML_H
#define SimG4Core_PhysicsLists_QGSBCMS_BERT_NOLEP1_EML_H
 
#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class QGSBCMS_BERT_NOLEP1_EML: public PhysicsList {

public:
  QGSBCMS_BERT_NOLEP1_EML(G4LogicalVolumeToDDLogicalPartMap& map, const HepPDT::ParticleDataTable * table_, sim::FieldBuilder *fieldBuilder_, const edm::ParameterSet & p);
};

#endif


