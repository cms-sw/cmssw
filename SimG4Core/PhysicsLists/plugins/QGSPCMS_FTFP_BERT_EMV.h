#ifndef SimG4Core_PhysicsLists_QGSPCMS_FTFP_BERT_EMV_H
#define SimG4Core_PhysicsLists_QGSPCMS_FTFP_BERT_EMV_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class QGSPCMS_FTFP_BERT_EMV : public PhysicsList {
public:
  QGSPCMS_FTFP_BERT_EMV(const edm::ParameterSet& p);
};

#endif
