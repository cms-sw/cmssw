#ifndef SimG4Core_PhysicsLists_QGSPCMS_BERT_EMLSNX_h
#define SimG4Core_PhysicsLists_QGSPCMS_BERT_EMLSNX_h 1

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class QGSPCMS_BERT_EMLSNX: public PhysicsList {

public:
  QGSPCMS_BERT_EMLSNX(G4LogicalVolumeToDDLogicalPartMap& map, const HepPDT::ParticleDataTable * table_, sim::FieldBuilder *fieldBuilder_, const edm::ParameterSet & p);

};

#endif



