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
// $Id: HSCPFilter.cc,v 1.5 2013/02/27 22:47:59 wmtan Exp $
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

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"


//
// class declaration
//

class HSCPFilter : public edm::EDFilter {
   public:
      explicit HSCPFilter(const edm::ParameterSet&);
      ~HSCPFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;
      bool filterFlag;
      edm::InputTag input_muon_collection, input_track_collection,input_dedx_collection;
      int ndedxHits;
      double dedxMin, dedxMaxLeft, trkPtMin,SAMuPtMin,etaMin,etaMax,chi2nMax,dxyMax,dzMax;
      
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
     filterFlag = iConfig.getParameter< bool >("filter");
     input_muon_collection = iConfig.getParameter< edm::InputTag >("inputMuonCollection");    
     input_track_collection = iConfig.getParameter< edm::InputTag >("inputTrackCollection");    
input_dedx_collection =  iConfig.getParameter< edm::InputTag >("inputDedxCollection");
     dedxMin = iConfig.getParameter< double >("dedxMin");
     dedxMaxLeft = iConfig.getParameter< double >("dedxMaxLeft");
     trkPtMin = iConfig.getParameter< double >("trkPtMin");
     etaMin =  iConfig.getParameter< double >("etaMin");
     etaMax =  iConfig.getParameter< double >("etaMax");
     ndedxHits = iConfig.getParameter< int >("ndedxHits");
     chi2nMax = iConfig.getParameter< double >("chi2nMax");
     dxyMax = iConfig.getParameter< double >("dxyMax");
     dzMax = iConfig.getParameter< double >("dzMax");
     SAMuPtMin = iConfig.getParameter< double >("SAMuPtMin");
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

  edm::Handle<reco::VertexCollection> recoVertexHandle;
   iEvent.getByLabel("offlinePrimaryVertices", recoVertexHandle);
   reco::VertexCollection recoVertex = *recoVertexHandle;

   if(!filterFlag) return true;

   if(recoVertex.size()<1) return false;

   using reco::MuonCollection;

   Handle<MuonCollection> muTracks;
   iEvent.getByLabel(input_muon_collection,muTracks);
   const reco::MuonCollection muonC = *(muTracks.product());
   for(unsigned int i=0; i<muonC.size(); i++){
      reco::MuonRef muon  = reco::MuonRef( muTracks, i );
      if(!muon->standAloneMuon().isNull()) {
         TrackRef SATrack = muon->standAloneMuon();
         if(SATrack->pt()>SAMuPtMin) return true;
      }

   }






   using reco::TrackCollection;
   Handle<TrackCollection> tkTracks;
   iEvent.getByLabel(input_track_collection,tkTracks);
   const reco::TrackCollection tkTC = *(tkTracks.product());

   Handle<ValueMap<DeDxData> >          dEdxTrackHandle;
   iEvent.getByLabel(input_dedx_collection, dEdxTrackHandle);
   const ValueMap<DeDxData> dEdxTrack = *dEdxTrackHandle.product();

   for(size_t i=0; i<tkTracks->size(); i++){

      reco::TrackRef trkRef = reco::TrackRef(tkTracks, i);
      
      
      if(trkRef->pt()>trkPtMin && trkRef->eta()<etaMax && trkRef->eta()>etaMin && trkRef->normalizedChi2()<chi2nMax){

           double dz  = trkRef->dz (recoVertex[0].position());
           double dxy = trkRef->dxy(recoVertex[0].position());
           double distancemin =sqrt(dxy*dxy+dz*dz);
           int closestvertex=0;
           for(unsigned int i=1;i<recoVertex.size();i++){
              dz  = trkRef->dz (recoVertex[i].position());
              dxy = trkRef->dxy(recoVertex[i].position());
              double distance = sqrt(dxy*dxy+dz*dz);
              if(distance < distancemin ){
                 distancemin = distance;
                 closestvertex=i;
              }
           }
           
           dz  = trkRef->dz (recoVertex[closestvertex].position());
           dxy = trkRef->dxy(recoVertex[closestvertex].position());
           
           if(fabs(dz)<dzMax && fabs(dxy)<dxyMax ){
              
             double dedx = dEdxTrack[trkRef].dEdx();
              int dedxnhits  = dEdxTrack[trkRef].numberOfMeasurements();
              if((dedx >dedxMin || dedx<dedxMaxLeft) && dedxnhits > ndedxHits) return true;
           }
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
