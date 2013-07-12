// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      TrivialExample
// 
/*

 Description: Trivial example to use get energy for a collection of ctfWithMaterialTracks

*/
//
// Original Author:  Dmytro Kovalskyi
// $Id: TrivialExample.cc,v 1.6 2007/04/13 03:09:28 dmytro Exp $

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/Handle.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"

class TrivialExample : public edm::EDAnalyzer {
 public:
   explicit TrivialExample(const edm::ParameterSet&);
   virtual ~TrivialExample(){}
   virtual void analyze (const edm::Event&, const edm::EventSetup&);

 private:
   TrackDetectorAssociator trackAssociator_;
   TrackAssociatorParameters parameters_;
};

TrivialExample::TrivialExample(const edm::ParameterSet& iConfig)
{
   // TrackAssociator parameters
   edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
   parameters_.loadParameters( parameters );
   trackAssociator_.useDefaultPropagator();
}

void TrivialExample::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // get reco tracks 
   edm::Handle<reco::TrackCollection> recoTracks;
   iEvent.getByLabel("ctfWithMaterialTracks", recoTracks);
   if (! recoTracks.isValid() ) throw cms::Exception("FatalError") << "No reco tracks were found\n";

   for(reco::TrackCollection::const_iterator recoTrack = recoTracks->begin();
       recoTrack != recoTracks->end(); ++recoTrack){
       
      if (recoTrack->pt() < 2) continue; // skip low Pt tracks

      
      TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, *recoTrack, parameters_);
      
      edm::LogVerbatim("TrackAssociator") << "\n-------------------------------------------------------\n Track (pt,eta,phi): " << 
	recoTrack->pt() << " , " << recoTrack->eta() << " , " << recoTrack->phi() ;
      edm::LogVerbatim("TrackAssociator") << "Ecal energy in crossed crystals based on RecHits: " << 
	info.crossedEnergy(TrackDetMatchInfo::EcalRecHits);
      edm::LogVerbatim("TrackAssociator") << "Ecal energy in 3x3 crystals based on RecHits: " << 
	info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 1);
      edm::LogVerbatim("TrackAssociator") << "Hcal energy in crossed towers based on RecHits: " << 
	info.crossedEnergy(TrackDetMatchInfo::HcalRecHits);
      edm::LogVerbatim("TrackAssociator") << "Hcal energy in 3x3 towers based on RecHits: " << 
	info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 1);
      edm::LogVerbatim("TrackAssociator") << "Number of muon segment matches: " << info.numberOfSegments();

   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrivialExample);
