#ifndef CaloCleanerMVA_h
#define CaloCleanerMVA_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleanerBase.h"


class CaloCleanerMVA : public CaloCleanerBase
{

  public:
    virtual ~CaloCleanerMVA(){};
    CaloCleanerMVA(const edm::ParameterSet& config);
    
    virtual std::string name() { return "Yay!!! I'm a CaloCleanerMVA!"; };
    virtual float cleanRH(DetId det, float energy);



};


#endif
