#ifndef CaloCleanerConst_h
#define CaloCleanerConst_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleanerBase.h"

#include "TauAnalysis/MCEmbeddingTools/interface/DetNaming.h"

#include <map>
#include <string>

class CaloCleanerConst : public CaloCleanerBase
{

  public:
    virtual ~CaloCleanerConst(){};
    CaloCleanerConst(const edm::ParameterSet& config);
    
    virtual std::string name() { return "Yay!!! I'm a CaloCleanerConst!"; };
    virtual float cleanRH(DetId det, float energy);

  private:
    edm::InputTag muons_;
    DetNaming detNaming_;

    std::map<std::string, float > values_;
};


#endif
