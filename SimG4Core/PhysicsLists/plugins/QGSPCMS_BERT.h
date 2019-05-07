#ifndef SimG4Core_PhysicsLists_QGSPCMS_BERT_H
#define SimG4Core_PhysicsLists_QGSPCMS_BERT_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Physics/interface/PhysicsList.h"

class QGSPCMS_BERT : public PhysicsList {

public:
  QGSPCMS_BERT(const edm::ParameterSet &p);
};

#endif
