#ifndef SimG4Core_PhysicsLists_FTFPCMS_BERT_EMH_H
#define SimG4Core_PhysicsLists_FTFPCMS_BERT_EMH_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class FTFPCMS_BERT_EMH : public PhysicsList {
public:
  FTFPCMS_BERT_EMH(const edm::ParameterSet& p);
};

#endif
