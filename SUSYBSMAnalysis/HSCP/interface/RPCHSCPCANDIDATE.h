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
#include <DataFormats/RPCRecHit/interface/RPCRecHit.h>
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

class  RPCHSCPCANDIDATE{
   public:
      explicit RPCHSCPCANDIDATE(edm::Event& iEvent, const edm::EventSetup& iSetup);
      bool found(){return foundvalue;}
      float eta(){return etavalue;}
      float phi(){return phivalue;}
      float beta(){return phivalue;}
        
   private:
      bool foundvalue;
      float phivalue;
      float etavalue;
      float betavalue;

      float etarange(float eta1,float eta2,float eta3);
      float dist(float phi1,float phi2);
      float dist3(float phi1,float phi2,float phi3);
};

