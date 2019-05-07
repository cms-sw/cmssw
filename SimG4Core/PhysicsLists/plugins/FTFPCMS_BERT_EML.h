#ifndef SimG4Core_PhysicsLists_FTFPCMS_BERT_EML_H
#define SimG4Core_PhysicsLists_FTFPCMS_BERT_EML_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Physics/interface/PhysicsList.h"

class FTFPCMS_BERT_EML : public PhysicsList {

public:
  FTFPCMS_BERT_EML(const edm::ParameterSet &p);
};

#endif
