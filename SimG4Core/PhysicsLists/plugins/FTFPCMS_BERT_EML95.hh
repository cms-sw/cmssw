#ifndef SimG4Core_PhysicsLists_FTFPCMS_BERT_EML95_H
#define SimG4Core_PhysicsLists_FTFPCMS_BERT_EML95_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class FTFPCMS_BERT_EML95: public PhysicsList {

public:
  FTFPCMS_BERT_EML95(G4LogicalVolumeToDDLogicalPartMap& map, const HepPDT::ParticleDataTable * table_, sim::FieldBuilder *fieldBuilder_, const edm::ParameterSet & p);
};

#endif



