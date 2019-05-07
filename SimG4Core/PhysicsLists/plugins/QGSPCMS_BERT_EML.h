#ifndef SimG4Core_PhysicsLists_QGSPCMS_BERT_EML_H
#define SimG4Core_PhysicsLists_QGSPCMS_BERT_EML_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Physics/interface/PhysicsList.h"

class QGSPCMS_BERT_EML : public PhysicsList {

public:
  QGSPCMS_BERT_EML(const edm::ParameterSet &p);
};

#endif
