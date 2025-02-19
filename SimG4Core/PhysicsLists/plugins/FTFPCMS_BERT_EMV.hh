#ifndef SimG4Core_PhysicsLists_FTFPCMS_BERT_EMV_H
#define SimG4Core_PhysicsLists_FTFPCMS_BERT_EMV_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class FTFPCMS_BERT_EMV: public PhysicsList {

public:
  FTFPCMS_BERT_EMV(G4LogicalVolumeToDDLogicalPartMap& map, const HepPDT::ParticleDataTable * table_, sim::FieldBuilder *fieldBuilder_, const edm::ParameterSet & p);
};

#endif



