#ifndef SimG4Core_PhysicsLists_FTFPCMS_BERT_EMY_H
#define SimG4Core_PhysicsLists_FTFPCMS_BERT_EMY_H

// FTFP_BERT_EMY is a standard Geant4 Physics List FTFP_BERT with Option3 
//               EM Physics configuration: forced more steps of e+- near 
//               geometry boundary. This configuration may be used for R&D of 
//               tracker and HGCal detector performnace. The similation
//               is expected to be approximately two times slower then 
//               with the CMS production Physics List

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class FTFPCMS_BERT_EMY: public PhysicsList {

public:
  FTFPCMS_BERT_EMY(const edm::ParameterSet & p);
};

#endif



