// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <DataFormats/MuonDetId/interface/MuonSubdetId.h>
#include <DataFormats/RPCRecHit/interface/RPCRecHit.h>

#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
#include <Geometry/RPCGeometry/interface/RPCRoll.h>
#include <DataFormats/RPCRecHit/interface/RPCRecHitCollection.h>

#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"


class  BetaCalculatorRPC{
   public:
      BetaCalculatorRPC(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);
      void  algo(const std::vector<susybsm::RPCHit4D>& HSCPRPCRecHits);
      void  addInfoToCandidate(susybsm::HSCParticle& candidate, const edm::Event& iEvent, const edm::EventSetup& iSetup);
      float beta(){return betavalue;}

   private:
      bool foundvalue;
      float phivalue;
      float etavalue;
      float betavalue;

      float etarange(float eta1,float eta2,float eta3);
      float dist(float phi1,float phi2);
      float dist3(float phi1,float phi2,float phi3);

      edm::EDGetTokenT<RPCRecHitCollection> rpcRecHitsToken;
};


