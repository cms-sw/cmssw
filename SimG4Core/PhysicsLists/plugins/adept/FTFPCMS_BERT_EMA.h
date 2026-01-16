#ifndef SimG4Core_PhysicsLists_FTFPCMS_BERT_EMA_H
#define SimG4Core_PhysicsLists_FTFPCMS_BERT_EMA_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class FTFPCMS_BERT_EMA : public PhysicsList {
public:
  FTFPCMS_BERT_EMA(const edm::ParameterSet& p);
};

#endif
