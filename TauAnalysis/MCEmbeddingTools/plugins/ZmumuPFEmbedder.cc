
/** \class ZmumuPFEmbedder
 *
 * Produce collections of PFCandidates and reco::Tracks
 * from which the two muons reconstructed in selected Z --> mu+ mu- events are removed
 * (later to be replaced by simulated tau decay products) 
 * 
 * \author Tomasz Maciej Frueboes
 *
 * \version $Revision: 1.8 $
 *
 * $Id: ZmumuPFEmbedder.cc,v 1.8 2012/02/13 17:33:04 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

class ZmumuPFEmbedder : public edm::EDProducer 
{
 public:
  explicit ZmumuPFEmbedder(const edm::ParameterSet&);
  ~ZmumuPFEmbedder() {}

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);
  void producePFCandColl(edm::Event&, const std::vector<reco::Particle::LorentzVector>*);
  void produceTrackColl(edm::Event&, const std::vector<reco::Particle::LorentzVector>*);
      
  edm::InputTag _tracks;
  edm::InputTag _selectedMuons;
  bool _useCombinedCandidate;
};

ZmumuPFEmbedder::ZmumuPFEmbedder(const edm::ParameterSet& iConfig)
  : _tracks(iConfig.getParameter<edm::InputTag>("tracks")),
    _selectedMuons(iConfig.getParameter<edm::InputTag>("selectedMuons")),
    _useCombinedCandidate(iConfig.getUntrackedParameter<bool>("useCombinedCandidate", false))
{
  produces<reco::TrackCollection>("tracks");
  produces<reco::PFCandidateCollection>("pfCands");
}

void
ZmumuPFEmbedder::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::vector<reco::Particle::LorentzVector> selMuonP4s;
   
  if ( _useCombinedCandidate ) {
    edm::Handle<reco::CompositeCandidateCollection> muonPairsHandle;
    if ( iEvent.getByLabel(_selectedMuons, muonPairsHandle) ) {
      if ( muonPairsHandle->size() >= 1 ) {
	const reco::CompositeCandidate& muonPair = muonPairsHandle->at(0); // TF: use only the first combined candidate
	for ( size_t idx = 0; idx < muonPair.numberOfDaughters(); ++idx ) { 
	  selMuonP4s.push_back(muonPair.daughter(idx)->p4());
	}
      }
    }
  } else {
    edm::Handle<reco::MuonCollection> selectedMuonsHandle;
    if ( iEvent.getByLabel(_selectedMuons, selectedMuonsHandle) ) {
      for ( size_t idx = 0; idx < selectedMuonsHandle->size(); ++idx ) {
	selMuonP4s.push_back(selectedMuonsHandle->at(idx).p4());
      }
    }
  }
  
  if ( selMuonP4s.size() <= 2 ) return; // not selected Z --> mu+ mu- event: do nothing
  
  producePFCandColl(iEvent, &selMuonP4s);
  produceTrackColl(iEvent, &selMuonP4s);
}

void ZmumuPFEmbedder::producePFCandColl(edm::Event & iEvent, const std::vector<reco::Particle::LorentzVector>* selMuonP4s)
{
//--- produce collection of PFCandidate excluding muons

  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  iEvent.getByLabel("particleFlow", pfCandidates);

  std::auto_ptr<reco::PFCandidateCollection> pfCandidates_woMuons(new reco::PFCandidateCollection());   
   
//--- iterate over list of reconstructed PFCandidates, 
//    add PFCandidate to output collection in case it does not correspond to any selected muon
  for ( reco::PFCandidateCollection::const_iterator pfCandidate = pfCandidates->begin();
	pfCandidate != pfCandidates->end(); ++pfCandidate ) {
    double dRmin = 1.e+3;
    for ( std::vector<reco::Particle::LorentzVector>::const_iterator selMuonP4 = selMuonP4s->begin();
	  selMuonP4 != selMuonP4s->end(); ++selMuonP4 ) {
      double dR = reco::deltaR(pfCandidate->p4(), *selMuonP4);
      if ( dR < dRmin ) dRmin = dR;
    }

    if ( dRmin < 1.e-3 ) continue; // it is a selected muon, do not copy
       
    pfCandidates_woMuons->push_back(*pfCandidate);
  }

  iEvent.put(pfCandidates_woMuons, "pfCands");
}

void ZmumuPFEmbedder::produceTrackColl(edm::Event & iEvent, const std::vector<reco::Particle::LorentzVector>* selMuonP4s)
{
//--- produce collection of reco::Tracks excluding muons

  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByLabel(_tracks, tracks);

  std::auto_ptr<reco::TrackCollection> tracks_woMuons(new reco::TrackCollection());

  for ( reco::TrackCollection::const_iterator track = tracks->begin();
	track != tracks->end(); ++track ) {
    double dRmin = 1.e+3;
    for ( std::vector<reco::Particle::LorentzVector>::const_iterator selMuonP4 = selMuonP4s->begin();
	  selMuonP4 != selMuonP4s->end(); ++selMuonP4 ) {
      double dR = reco::deltaR(track->eta(), track->phi(), selMuonP4->eta(), selMuonP4->phi());
      if ( dR < dRmin ) dRmin = dR;
    }

    if ( dRmin < 1.e-4 ) continue; // it is a selected muon, do not copy
  }

  iEvent.put(tracks_woMuons, "tracks");
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZmumuPFEmbedder);
