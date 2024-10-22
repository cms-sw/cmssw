#ifndef SimG4Core_PhysicsLists_QGSPCMS_FTFP_BERT_H
#define SimG4Core_PhysicsLists_QGSPCMS_FTFP_BERT_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class QGSPCMS_FTFP_BERT : public PhysicsList {
public:
  QGSPCMS_FTFP_BERT(const edm::ParameterSet& p);
};

#endif
