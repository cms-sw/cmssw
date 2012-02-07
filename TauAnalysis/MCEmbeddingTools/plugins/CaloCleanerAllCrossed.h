#ifndef CaloCleanerAllCrossed_h
#define CaloCleanerAllCrossed_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleanerBase.h"


class CaloCleanerAllCrossed : public CaloCleanerBase
{

  public:
    virtual ~CaloCleanerAllCrossed(){};
    CaloCleanerAllCrossed(const edm::ParameterSet& config);
    
    virtual std::string name() { return "Yay!!! I'm a CaloCleanerAllCrossed!"; };
    virtual float cleanRH(DetId det, float energy);




};


#endif
