#ifndef SimG4Core_PhysicsLists_QGSBCMS_BERT_NOLEP2_H
#define SimG4Core_PhysicsLists_QGSBCMS_BERT_NOLEP2_H
 
#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class QGSBCMS_BERT_NOLEP2: public PhysicsList {

public:
  QGSBCMS_BERT_NOLEP2(G4LogicalVolumeToDDLogicalPartMap& map, const HepPDT::ParticleDataTable * table_, sim::FieldBuilder *fieldBuilder_, const edm::ParameterSet & p);
};

#endif


