// -*- C++ -*-
//
// Package:    ProduceIsolationMap
// Class:      ProduceIsolationMap
//
/*\class ProduceIsolationMap ProduceIsolationMap.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Loic Quertenmont
//         Created:  Wed Nov 10 16:41:46 CDT 2010
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfo.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCPIsolation.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <iostream>

//
// class declaration
//

using namespace susybsm;
using namespace edm;

class ProduceIsolationMap : public edm::EDProducer {
   public:
      explicit ProduceIsolationMap(const edm::ParameterSet&);
      ~ProduceIsolationMap();
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
   private:
      edm::EDGetTokenT<reco::TrackCollection> TKToken_;
      edm::EDGetTokenT<reco::TrackCollection> inputCollectionToken_;
      double  TKIsolationPtcut_;
      double  IsolationConeDR_;
      TrackDetectorAssociator trackAssociator_;
      TrackAssociatorParameters parameters_;
      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
ProduceIsolationMap::ProduceIsolationMap(const edm::ParameterSet& iConfig)
{
   TKToken_          = consumes<reco::TrackCollection>(iConfig.getParameter< edm::InputTag > ("TKLabel"));
   inputCollectionToken_  = consumes<reco::TrackCollection>(iConfig.getParameter< edm::InputTag > ("inputCollection"));
   TKIsolationPtcut_ = iConfig.getParameter< double >        ("TkIsolationPtCut");
   IsolationConeDR_  = iConfig.getParameter< double >        ("IsolationConeDR");


   // TrackAssociator parameters
   edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
   parameters_.loadParameters( parameters );
   trackAssociator_.useDefaultPropagator();

   //register your products
    produces<ValueMap<HSCPIsolation> >();
}


ProduceIsolationMap::~ProduceIsolationMap()
{
}

void
ProduceIsolationMap::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

   using reco::TrackCollection;

   Handle<TrackCollection> TKHandle;
   iEvent.getByToken(TKToken_,TKHandle);
   if(!TKHandle.isValid() ){  edm::LogError("ProduceIsolationMap") << "TK Tracks collection not found";    return;   }

   //Create empty output collections
   auto_ptr<ValueMap<HSCPIsolation> > trackHSCPIsolMap(new ValueMap<HSCPIsolation> );
   ValueMap<HSCPIsolation>::Filler    filler(*trackHSCPIsolMap);

   //loop through tracks.
   Handle<TrackCollection> tkTracks;
   iEvent.getByToken(inputCollectionToken_,tkTracks);
   std::vector<HSCPIsolation> IsolationInfoColl(tkTracks->size());

   int TkIndex=0;
   for(TrackCollection::const_iterator itTrack = tkTracks->begin(); itTrack != tkTracks->end(); ++itTrack, TkIndex++) {
      TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, *itTrack, parameters_, TrackDetectorAssociator::InsideOut);


      if(info.ecalRecHits.size()>0){IsolationInfoColl[TkIndex].Set_ECAL_Energy(info.coneEnergy(IsolationConeDR_, TrackDetMatchInfo::EcalRecHits));}
      if(info.hcalRecHits.size()>0){IsolationInfoColl[TkIndex].Set_HCAL_Energy(info.coneEnergy(IsolationConeDR_, TrackDetMatchInfo::HcalRecHits));}
//      if(info.hcalRecHits.size()>0){IsolationInfoColl[TkIndex].Set_HCAL_Energy(info.hcalConeEnergy());}
//      if(info.ecalRecHits.size()>0){IsolationInfoColl[TkIndex].Set_ECAL_Energy(info.ecalConeEnergy());}

      double SumPt       = 0;
      double Count       = 0;
      double CountHighPt = 0;
      for(TrackCollection::const_iterator itTrack2 = TKHandle->begin(); itTrack2 != TKHandle->end(); ++itTrack2){
         if(fabs(itTrack->pt()-itTrack2->pt())<0.1 && fabs(itTrack->eta()-itTrack2->eta())<0.05)continue;
         float dR = deltaR(itTrack->momentum(), itTrack2->momentum());
         if(dR>IsolationConeDR_)continue;
         SumPt+= itTrack2->pt();
         Count++;
         if(itTrack2->pt()<TKIsolationPtcut_)continue;
         CountHighPt++;
      }
      IsolationInfoColl[TkIndex].Set_TK_CountHighPt(CountHighPt);
      IsolationInfoColl[TkIndex].Set_TK_Count      (Count);
      IsolationInfoColl[TkIndex].Set_TK_SumEt      (SumPt);
   }

   filler.insert(tkTracks, IsolationInfoColl.begin(), IsolationInfoColl.end());
   filler.fill();
   iEvent.put(trackHSCPIsolMap);
}
//define this as a plug-in
DEFINE_FWK_MODULE(ProduceIsolationMap);
