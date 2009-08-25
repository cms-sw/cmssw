// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h" 
#include "FWCore/Framework/interface/ESHandle.h"

#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"

class  BetaFromRPC{
   public:
      BetaFromRPC(std::vector<susybsm::RPCHit4D> HSCPRPCRecHits);
      float beta(){return betavalue;}
        
   private:
      bool foundvalue;
      float phivalue;
      float etavalue;
      float betavalue;

      float etarange(float eta1,float eta2,float eta3);
      float dist(float phi1,float phi2);
      float dist3(float phi1,float phi2,float phi3);
};
