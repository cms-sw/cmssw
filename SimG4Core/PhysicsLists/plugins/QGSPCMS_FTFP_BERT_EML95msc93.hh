#ifndef SimG4Core_PhysicsLists_QGSPCMS_FTFP_BERT_EML95msc93_H
#define SimG4Core_PhysicsLists_QGSPCMS_FTFP_BERT_EML95msc93_H
 
#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class QGSPCMS_FTFP_BERT_EML95msc93: public PhysicsList {

public:
  QGSPCMS_FTFP_BERT_EML95msc93(G4LogicalVolumeToDDLogicalPartMap& map, const HepPDT::ParticleDataTable * table_, sim::FieldBuilder *fieldBuilder_, const edm::ParameterSet & p);
};

#endif


