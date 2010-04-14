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
// $Id: HSCP_Producer.cc,v 1.2 2010/04/13 16:15:31 querten Exp $
//
//

// user include files
#include "SUSYBSMAnalysis/HSCP/interface/HSCP_Producer.h"

HSCP_Producer::HSCP_Producer(const edm::ParameterSet& iConfig) {
  using namespace edm;
  using namespace std;

  // the input collections
  m_trackTag      = iConfig.getParameter<edm::InputTag>("tracks");
  m_muonsTag      = iConfig.getParameter<edm::InputTag>("muons");

  useBetaFromTk   = iConfig.getParameter<bool>    ("useBetaFromTk"  );
  useBetaFromMuon = iConfig.getParameter<bool>    ("useBetaFromMuon");
  useBetaFromRpc  = iConfig.getParameter<bool>    ("useBetaFromRpc" );
  useBetaFromEcal = iConfig.getParameter<bool>    ("useBetaFromEcal");

  // the parameters
  minTkP          = iConfig.getParameter<double>  ("minTkP");       // 30
  maxTkChi2       = iConfig.getParameter<double>  ("maxTkChi2");    // 5
  minTkHits       = iConfig.getParameter<uint32_t>("minTkHits");    // 9
  minMuP          = iConfig.getParameter<double>  ("minMuP");       // 30
  minDR           = iConfig.getParameter<double>  ("minDR");        // 0.1
  maxInvPtDiff    = iConfig.getParameter<double>  ("maxInvPtDiff"); // 0.005

  maxTkBeta       = iConfig.getParameter<double>  ("maxTkBeta");    // 0.9; 
  maxMuBeta       = iConfig.getParameter<double>  ("maxMuBeta");    // 0.9; 

  if(useBetaFromTk  )beta_calculator_TK   = new Beta_Calculator_TK  (iConfig);
  if(useBetaFromMuon)beta_calculator_MUON = new Beta_Calculator_MUON(iConfig);
  if(useBetaFromRpc )beta_calculator_RPC  = new Beta_Calculator_RPC (iConfig);
  if(useBetaFromEcal)beta_calculator_ECAL = new Beta_Calculator_ECAL(iConfig);

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

  std::cout<<"TEST1\n";

  // information from the muons
  edm::Handle<reco::MuonCollection> muonCollectionHandle;
  iEvent.getByLabel("muons",muonCollectionHandle);

  std::cout<<"TEST2\n";

  // information from the tracks
  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByLabel(m_trackTag,trackCollectionHandle);

  std::cout<<"TEST3\n";


  // creates the output collection
  susybsm::HSCParticleCollection* hscp = new susybsm::HSCParticleCollection; 
  std::auto_ptr<susybsm::HSCParticleCollection> result(hscp);

  std::cout<<"TEST4\n";


  // Fill the output collection with HSCP Candidate (the candiate only contains ref to muon AND/OR track object)
  *hscp = getHSCPSeedCollection(trackCollectionHandle, muonCollectionHandle);


  std::cout<<"TEST5\n";


  // compute the TRACKER contribution
  if(useBetaFromTk){
  for(susybsm::HSCParticleCollection::iterator hscpcandidate = hscp->begin(); hscpcandidate < hscp->end(); ++hscpcandidate) {
    beta_calculator_TK->addInfoToCandidate(*hscpcandidate,  iEvent,iSetup);
  }}

  std::cout<<"TEST6\n";


  // compute the MUON contribution
  if(useBetaFromMuon){
  for(susybsm::HSCParticleCollection::iterator hscpcandidate = hscp->begin(); hscpcandidate < hscp->end(); ++hscpcandidate) {
    beta_calculator_MUON->addInfoToCandidate(*hscpcandidate,  iEvent,iSetup);
  }}

  std::cout<<"TEST7\n";

  // compute the RPC contribution
  if(useBetaFromRpc){
  for(susybsm::HSCParticleCollection::iterator hscpcandidate = hscp->begin(); hscpcandidate < hscp->end(); ++hscpcandidate) {
      beta_calculator_RPC->addInfoToCandidate(*hscpcandidate, iSetup);
  }}

  std::cout<<"TEST8\n";

  // compute the ECAL contribution
  if(useBetaFromEcal){
  for(susybsm::HSCParticleCollection::iterator hscpcandidate = hscp->begin(); hscpcandidate < hscp->end(); ++hscpcandidate) {
    beta_calculator_ECAL->addInfoToCandidate(*hscpcandidate,trackCollectionHandle,iEvent,iSetup);
  }}

  std::cout<<"TEST9\n";

  // output result
  iEvent.put(result); 

  std::cout<<"TEST10\n";

}

// ------------ method called once each job just before starting event loop  ------------
void 
HSCP_Producer::beginJob() {
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HSCP_Producer::endJob() {
}

std::vector<HSCParticle> HSCP_Producer::getHSCPSeedCollection(edm::Handle<reco::TrackCollection>& trackCollectionHandle,  edm::Handle<reco::MuonCollection>& muonCollectionHandle)
{
  std::cout<<"MATCH1\n";

   std::vector<HSCParticle> HSCPCollection;

  std::cout<<"MATCH2\n";


   // Store a local vector of track ref (that can be modified if matching)
   std::vector<reco::TrackRef> tracks;
   for(unsigned int i=0; i<trackCollectionHandle->size(); i++){
      TrackRef track = reco::TrackRef( trackCollectionHandle, i );
      if(track->p()<minTkP || (track->chi2()/track->ndof())>maxTkChi2 || track->found()<minTkHits)continue;
      tracks.push_back( track );
   }

  std::cout<<"MATCH3\n";


   // Loop on muons and create Muon HSCP Candidate
   for(unsigned int m=0; m<muonCollectionHandle->size(); m++){
      reco::MuonRef muon  = reco::MuonRef( muonCollectionHandle, m );
      if(muon->p()<minMuP)continue;

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
        std::cout<<"MATCH_E1\n";
        candidate.setTrack(tracks[found]);
        std::cout<<"MATCH_E2\n";
        tracks.erase(tracks.begin()+found);
        std::cout<<"MATCH_E3\n";
      }
      HSCPCollection.push_back(candidate);
   }

  std::cout<<"MATCH4\n";

   // Loop on tracks not matching muon and create Track HSCP Candidate
   for(unsigned int i=0; i<tracks.size(); i++){
      HSCParticle candidate;
      candidate.setTrack(tracks[i]);
      HSCPCollection.push_back(candidate);
   }

  std::cout<<"MATCH5\n";

   return HSCPCollection;
}














//define this as a plug-in
DEFINE_FWK_MODULE(HSCP_Producer);



















