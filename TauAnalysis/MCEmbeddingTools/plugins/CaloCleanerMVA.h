#ifndef CaloCleanerMVA_h
#define CaloCleanerMVA_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleanerBase.h"

#include "TMVA/Reader.h"
#include "TauAnalysis/MCEmbeddingTools/interface/DetNaming.h"

#include <map>
#include <string>

class CaloCleanerMVA : public CaloCleanerBase
{

  public:
    virtual ~CaloCleanerMVA(){};
    CaloCleanerMVA(const edm::ParameterSet& config);
    
    virtual std::string name() { return "Yay!!! I'm a CaloCleanerMVA!"; };
    virtual float cleanRH(DetId det, float energy);

  private:
    edm::InputTag muons_;
   
    std::map<std::string, TMVA::Reader *>  readers_;
    std::map<std::string, std::string >    methods_;

    float pt, p, eta, phi, len, charge;


    DetNaming detNaming_;

};


#endif
