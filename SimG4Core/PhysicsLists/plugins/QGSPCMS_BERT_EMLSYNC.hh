#ifndef SimG4Core_PhysicsLists_QGSPCMS_BERT_EMLSYNC_H
#define SimG4Core_PhysicsLists_QGSPCMS_BERT_EMLSYNC_H
 
#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class QGSPCMS_BERT_EMLSYNC: public PhysicsList {

public:
  QGSPCMS_BERT_EMLSYNC(G4LogicalVolumeToDDLogicalPartMap& map, 
		       const edm::ParameterSet & p);
};

#endif


