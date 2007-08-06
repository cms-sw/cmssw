#ifndef SimG4Core_PhysicsLists_QGSPCMS_BERT_H
#define SimG4Core_PhysicsLists_QGSPCMS_BERT_H
 
#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class QGSPCMS_BERT: public PhysicsList {

public:
  QGSPCMS_BERT(G4LogicalVolumeToDDLogicalPartMap& map, 
	       const edm::ParameterSet & p);
};

#endif


