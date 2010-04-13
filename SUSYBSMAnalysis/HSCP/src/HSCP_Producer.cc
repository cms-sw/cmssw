// -*- C++ -*-
//
// Package:    HSCP_Producer
// Class:      HSCP_Producer
// 
/**\class HSCP_Producer HSCP_Producer.cc SUSYBSMAnalysis/HSCP_Producer/src/HSCP_Producer.cc

 Description: Producer for HSCP candidates, merging tracker dt information and rpc information

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Rizzi Andrea
// Reworked and Ported to CMSSW_3_0_0 by Christophe Delaere
//         Created:  Wed Oct 10 12:01:28 CEST 2007
// $Id: HSCP_Producer.cc,v 1.1 2010/04/12 15:42:01 querten Exp $
//
//

// user include files
#include "SUSYBSMAnalysis/HSCP/interface/HSCP_Producer.h"

HSCP_Producer::HSCP_Producer(const edm::ParameterSet& iConfig) {
  using namespace edm;
  using namespace std;

  // the input collections
  m_trackTag = iConfig.getParameter<edm::InputTag>("tracks");
  m_muonsTag = iConfig.getParameter<edm::InputTag>("muons");

  // the parameters
  minTkP=iConfig.getParameter<double>("minTkP"); // 30
  maxTkBeta=iConfig.getParameter<double>("maxTkBeta"); //0.9;
  minDtP=iConfig.getParameter<double>("minDtP"); //30
  minDR=iConfig.getParameter<double>("minDR"); //0.1
  maxInvPtDiff=iConfig.getParameter<double>("maxInvPtDiff"); //0.005
  maxChi2=iConfig.getParameter<double>("maxTkChi2"); //5
  minTkHits=iConfig.getParameter<uint32_t>("minTkHits"); //9

  beta_calculator_RPC  = new Beta_Calculator_RPC (iConfig);
  beta_calculator_ECAL = new Beta_Calculator_ECAL(iConfig);


  // what I produce
  produces<susybsm::HSCParticleCollection >();
}

HSCP_Producer::~HSCP_Producer() {
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
HSCP_Producer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;
  using namespace reco;
  using namespace std;
  using namespace susybsm;


  // information from the muons
  edm::Handle<reco::MuonCollection> muonCollectionHandle;
  iEvent.getByLabel("muons",muonCollectionHandle);

  // information from the tracks
  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByLabel(m_trackTag,trackCollectionHandle);


  // creates the output collection
  susybsm::HSCParticleCollection* hscp = new susybsm::HSCParticleCollection; 
  std::auto_ptr<susybsm::HSCParticleCollection> result(hscp);

  // Fill the output collection with HSCP Candidate (the candiate only contains ref to muon AND/OR track object)
  *hscp = getHSCPSeedCollection(trackCollectionHandle, muonCollectionHandle);


  // compute the TRACKER contribution
  for(susybsm::HSCParticleCollection::iterator hscpcandidate = hscp->begin(); hscpcandidate < hscp->end(); ++hscpcandidate) {
    beta_calculator_TK->addInfoToCandidate(*hscpcandidate,  iEvent,iSetup);
  }

  // compute the MUON contribution
  for(susybsm::HSCParticleCollection::iterator hscpcandidate = hscp->begin(); hscpcandidate < hscp->end(); ++hscpcandidate) {
    beta_calculator_MUON->addInfoToCandidate(*hscpcandidate,  iEvent,iSetup);
  }

  // compute the RPC contribution
  for(susybsm::HSCParticleCollection::iterator hscpcandidate = hscp->begin(); hscpcandidate < hscp->end(); ++hscpcandidate) {
      beta_calculator_RPC->addInfoToCandidate(*hscpcandidate, iSetup);
  }

  // compute the ECAL contribution
  for(susybsm::HSCParticleCollection::iterator hscpcandidate = hscp->begin(); hscpcandidate < hscp->end(); ++hscpcandidate) {
    beta_calculator_ECAL->addInfoToCandidate(*hscpcandidate,trackCollectionHandle,iEvent,iSetup);
  }

  // output result
  iEvent.put(result); 
}

// ------------ method called once each job just before starting event loop  ------------
void 
HSCP_Producer::beginJob(const edm::EventSetup& iSetup) {
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HSCP_Producer::endJob() {
}

std::vector<HSCParticle> HSCP_Producer::getHSCPSeedCollection(edm::Handle<reco::TrackCollection>& trackCollectionHandle,  edm::Handle<reco::MuonCollection>& muonCollectionHandle)
{
   std::vector<HSCParticle> HSCPCollection;

   // Store a local vector of track ref (that can be modified if matching)
   std::vector<reco::TrackRef> tracks;
   for(unsigned int i=0; i<trackCollectionHandle->size(); i++){
      tracks.push_back(reco::TrackRef( trackCollectionHandle, i ) );
   }

   // Loop on muons and create Muon HSCP Candidate
   for(unsigned int m=0; m<muonCollectionHandle->size(); m++){
      reco::MuonRef muon  = reco::MuonRef( muonCollectionHandle, m );

      // Check if the muon match any track in order to create a Muon+Track HSCP Candidate
      float dRMin=1000; int found = -1;
      for(unsigned int t=0; t<tracks.size();t++) {
         reco::TrackRef track  = tracks[t];
         if( fabs( (1.0/muon->pt())-(1.0/track->pt())) > maxInvPtDiff) continue;
         float dR = deltaR(muon->momentum(), track->momentum());
         if(dR <= minDR && dR < dRMin){ dRMin=dR; found = t;}
      }

      HSCParticle candidate;
      candidate.setMuon(muon);
      if(found>=0){
        candidate.setTrack(tracks[found]);
        tracks.erase(tracks.begin()+found);
      }
      HSCPCollection.push_back(candidate);
   }

   // Loop on tracks not matching muon and create Track HSCP Candidate
   for(unsigned int i=0; i<trackCollectionHandle->size(); i++){
      HSCParticle candidate;
      candidate.setTrack(tracks[i]);
      HSCPCollection.push_back(candidate);
   }

   return HSCPCollection;
}














//define this as a plug-in
DEFINE_FWK_MODULE(HSCP_Producer);



















