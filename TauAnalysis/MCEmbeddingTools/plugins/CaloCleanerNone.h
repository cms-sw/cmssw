#ifndef CaloCleanerNone_h
#define CaloCleanerNone_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleanerBase.h"


class CaloCleanerNone : public CaloCleanerBase
{

  public:
    virtual ~CaloCleanerNone(){};
    CaloCleanerNone(const edm::ParameterSet& config);
    
    virtual std::string name() { return "Yay!!! I'm a CaloCleanerNone!"; };
    virtual float cleanRH(DetId det, float energy);




};


#endif
