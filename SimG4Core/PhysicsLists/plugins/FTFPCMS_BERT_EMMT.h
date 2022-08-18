#ifndef SimG4Core_PhysicsLists_FTFPCMS_BERT_EMMT_H
#define SimG4Core_PhysicsLists_FTFPCMS_BERT_EMMT_H

#include "G4Version.hh"
#if G4VERSION_NUMBER >= 1100

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class FTFPCMS_BERT_EMMT : public PhysicsList {
public:
  FTFPCMS_BERT_EMMT(const edm::ParameterSet& p);
};

#endif

#endif
