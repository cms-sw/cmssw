#ifndef SimG4Core_PhysicsLists_QGSPCMS_BERT_EMM_H
#define SimG4Core_PhysicsLists_QGSPCMS_BERT_EMM_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class QGSPCMS_BERT_EMM : public PhysicsList {
public:
  QGSPCMS_BERT_EMM(const edm::ParameterSet& p);
};

#endif
