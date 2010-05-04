// -*- C++ -*-
//
// Package:    HSCPFilter
// Class:      HSCPFilter
// 
/**\class HSCPFilter HSCPFilter.cc HSCPFilter/HSCPFilter/src/HSCPFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jie Chen
//         Created:  Thu Apr 29 16:32:10 CDT 2010
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"


//
// class declaration
//

class HSCPFilter : public edm::EDFilter {
   public:
      explicit HSCPFilter(const edm::ParameterSet&);
      ~HSCPFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      edm::InputTag input_track_collection,input_dedx_collection;
      int ndedxHits;
      double dedxMin, trkPtMin,etaMin,etaMax,chi2nMax,d0Max,dzMax;
      
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
HSCPFilter::HSCPFilter(const edm::ParameterSet& iConfig)
{
     input_track_collection = iConfig.getParameter< edm::InputTag >("inputTrackCollection");    
input_dedx_collection =  iConfig.getParameter< edm::InputTag >("inputDedxCollection");
     dedxMin = iConfig.getParameter< double >("dedxMin");
     trkPtMin = iConfig.getParameter< double >("trkPtMin");
     etaMin =  iConfig.getParameter< double >("etaMin");
     etaMax =  iConfig.getParameter< double >("etaMax");
     ndedxHits = iConfig.getParameter< int >("ndedxHits");
     chi2nMax = iConfig.getParameter< double >("chi2nMax");
     d0Max = iConfig.getParameter< double >("d0Max");
     dzMax = iConfig.getParameter< double >("dzMax");

}


HSCPFilter::~HSCPFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HSCPFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif

   using namespace edm;
   using namespace reco;


   using reco::TrackCollection;
   Handle<TrackCollection> tkTracks;
   iEvent.getByLabel(input_track_collection,tkTracks);
   const reco::TrackCollection tkTC = *(tkTracks.product());

   Handle<ValueMap<DeDxData> >          dEdxTrackHandle;
   iEvent.getByLabel(input_dedx_collection, dEdxTrackHandle);
   const ValueMap<DeDxData> dEdxTrack = *dEdxTrackHandle.product();

   for(TrackCollection::const_iterator itTrack = tkTracks->begin();
       itTrack != tkTracks->end();                      
       ++itTrack) {
        int iTrack = itTrack - tkTracks->begin();
        if(itTrack->pt()>trkPtMin && itTrack->eta()<etaMax && itTrack->eta()>etaMin && itTrack->normalizedChi2()<chi2nMax && fabs(itTrack->dz())<dzMax && fabs(itTrack->d0())<d0Max ){
           reco::TrackRef track = reco::TrackRef(tkTracks, iTrack);           
           double dedx = dEdxTrack[track].dEdx();
           int dedxnhits  = dEdxTrack[track].numberOfMeasurements();
           if(dedx >dedxMin && dedxnhits > ndedxHits) return true;
        }
   }
   return false;

}

// ------------ method called once each job just before starting event loop  ------------
void 
HSCPFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HSCPFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HSCPFilter);
