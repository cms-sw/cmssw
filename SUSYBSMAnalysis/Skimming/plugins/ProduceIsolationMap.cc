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
// $Id: ProduceIsolationMap.cc,v 1.1 2010/04/14 14:30:38 jiechen Exp $
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
      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      edm::InputTag TKLabel_;
      edm::InputTag EBrecHitsLabel_;
      edm::InputTag EErecHitsLabel_;
      edm::InputTag HCALrecHitsLabel_;
      edm::InputTag inputCollection_;
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
   TKLabel_          = iConfig.getParameter< edm::InputTag > ("TKLabel");
   EBrecHitsLabel_   = iConfig.getParameter< edm::InputTag > ("EBrecHitsLabel");
   EErecHitsLabel_   = iConfig.getParameter< edm::InputTag > ("EErecHitsLabel");
   HCALrecHitsLabel_ = iConfig.getParameter< edm::InputTag > ("HCALrecHitsLabel");
   inputCollection_  = iConfig.getParameter< edm::InputTag > ("inputCollection");
   TKIsolationPtcut_ = iConfig.getParameter< double >        ("TkIsolationPtCut");
   IsolationConeDR_  = iConfig.getParameter< double >        ("IsolationConeDR");


   // TrackAssociator parameters
   edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
   parameters_.loadParameters( parameters );
   trackAssociator_.useDefaultPropagator();


   parameters_.theEBRecHitCollectionLabel   = EBrecHitsLabel_;
   parameters_.theEERecHitCollectionLabel   = EErecHitsLabel_;
   parameters_.theHBHERecHitCollectionLabel = HCALrecHitsLabel_;
//   parameters_.dREcal                       = IsolationConeDR_;
//   parameters_.dRHcal                       = IsolationConeDR_;
//   parameters_.dRMuon                       = IsolationConeDR_;
  
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
/*
   Handle<HBHERecHitCollection> HCALHitsHandle;
   iEvent.getByLabel(HCALrecHitsLabel_,HCALHitsHandle);
   if(!HCALHitsHandle.isValid() ){  edm::LogError("ProduceIsolationMap") << "HCAL RecHit collection not found";    return;   }
   
   Handle<EcalRecHitCollection> EBHitsHandle;
   iEvent.getByLabel(EBrecHitsLabel_,EBHitsHandle);
   if(!EBHitsHandle.isValid() ){  edm::LogError("ProduceIsolationMap") << "EB RecHit collection not found";    return;   }

   Handle<EcalRecHitCollection> EEHitsHandle;
   iEvent.getByLabel(EErecHitsLabel_,EEHitsHandle);
   if(!EEHitsHandle.isValid() ){  edm::LogError("ProduceIsolationMap") << "EE RecHit collection not found";    return;   }
*/
   Handle<TrackCollection> TKHandle;
   iEvent.getByLabel(TKLabel_,TKHandle);
   if(!TKHandle.isValid() ){  edm::LogError("ProduceIsolationMap") << "TK Tracks collection not found";    return;   }

   //Create empty output collections
   auto_ptr<ValueMap<HSCPIsolation> > trackHSCPIsolMap(new ValueMap<HSCPIsolation> );  
   ValueMap<HSCPIsolation>::Filler    filler(*trackHSCPIsolMap);
  
   //loop through tracks. 
   Handle<TrackCollection> tkTracks;
   iEvent.getByLabel(inputCollection_,tkTracks);
   std::vector<HSCPIsolation> IsolationInfoColl(tkTracks->size());

//   parameters_.dREcal = IsolationConeDR_;   parameters_.dRHcal = IsolationConeDR_;   parameters_.dRMuon = IsolationConeDR_;

   int TkIndex=0;
   for(TrackCollection::const_iterator itTrack = tkTracks->begin(); itTrack != tkTracks->end(); ++itTrack, TkIndex++) {
//    if(itTrack->pt()>ptcut_){

      TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, *itTrack, parameters_, TrackDetectorAssociator::InsideOut);

      //  cout<<"isolation cone "<<IsolationConeDR_<<endl;
      if(info.hcalRecHits.size()>0){IsolationInfoColl[TkIndex].Set_HCAL_Energy(info.coneEnergy(IsolationConeDR_,TrackDetMatchInfo::HcalRecHits));}
      if(info.ecalRecHits.size()>0){IsolationInfoColl[TkIndex].Set_ECAL_Energy(info.coneEnergy(IsolationConeDR_,TrackDetMatchInfo::EcalRecHits));}

      double SumPt = 0;
      double Count = 0;
      for(TrackCollection::const_iterator itTrack2 = TKHandle->begin(); itTrack2 != TKHandle->end(); ++itTrack2){
         if(itTrack2->pt()<TKIsolationPtcut_) Count++;
         if(fabs(itTrack->pt()-itTrack2->pt())<0.001 && fabs(itTrack->eta()-itTrack2->eta())<0.001)continue;
         float dR = deltaR(itTrack->momentum(), itTrack2->momentum());
         if(dR<IsolationConeDR_){SumPt+= itTrack2->pt();}
      }
      IsolationInfoColl[TkIndex].Set_TK_Count(Count);
      IsolationInfoColl[TkIndex].Set_TK_SumEt(SumPt);
      
//      cout<<"trk "<<TkIndex<<" p "<<itTrack->p() <<" Eecal "<<IsolationInfoColl[TkIndex].Get_ECAL_Energy() <<" ecal3by3 "<<info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 1)<<" Hcal "<<IsolationInfoColl[TkIndex].Get_HCAL_Energy()<<" Hcal3by3 " << info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 1) <<" eop "<<(IsolationInfoColl[TkIndex].Get_ECAL_Energy()+IsolationInfoColl[TkIndex].Get_HCAL_Energy())/itTrack->p() <<" trkiso "<<SumPt/itTrack->pt() <<endl;
   }

   filler.insert(tkTracks, IsolationInfoColl.begin(), IsolationInfoColl.end());
   filler.fill();
   iEvent.put(trackHSCPIsolMap); 
}
//define this as a plug-in
DEFINE_FWK_MODULE(ProduceIsolationMap);
