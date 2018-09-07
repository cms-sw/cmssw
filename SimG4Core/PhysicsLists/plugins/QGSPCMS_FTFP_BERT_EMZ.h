#ifndef SimG4Core_PhysicsLists_QGSPCMS_FTFP_BERT_EMZ_H
#define SimG4Core_PhysicsLists_QGSPCMS_FTFP_BERT_EMZ_H 1

// QGSP_FTFP_BERT_EMZ is a standard Geant4 Physics List FTFP_BERT with Option4
//                    EM Physics configuration: used GS multiple scattering 
//                    model for e+- below 100 MeV. Very precised simulation 
//                    for thin layers. This configuration may be used for R&D 
//                    of tracker and HGCal detector performnace. The similation
//                    is expected to be approximately two times slower then 
//                    with the CMS production Physics List
 
#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class QGSPCMS_FTFP_BERT_EMZ: public PhysicsList {

public:
  QGSPCMS_FTFP_BERT_EMZ(const edm::ParameterSet & p);
};

#endif


