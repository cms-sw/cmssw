#ifndef SimG4Core_PhysicsLists_FTFPCMS_BERT_H
#define SimG4Core_PhysicsLists_FTFPCMS_BERT_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class FTFPCMS_BERT: public PhysicsList {

public:
  FTFPCMS_BERT(const edm::ParameterSet & p);
};

#endif



