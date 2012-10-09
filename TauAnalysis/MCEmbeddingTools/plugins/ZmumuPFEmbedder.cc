
/** \class ZmumuPFEmbedder
 *
 * Produce collections of PFCandidates and reco::Tracks
 * from which the two muons reconstructed in selected Z --> mu+ mu- events are removed
 * (later to be replaced by simulated tau decay products) 
 * 
 * \author Tomasz Maciej Frueboes
 *
 * \version $Revision: 1.13 $
 *
 * $Id: ZmumuPFEmbedder.cc,v 1.13 2012/10/07 13:09:35 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
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
      
  edm::InputTag srcTracks_;
  edm::InputTag srcPFCandidates_;
  edm::InputTag srcSelectedMuons_;
};

ZmumuPFEmbedder::ZmumuPFEmbedder(const edm::ParameterSet& cfg)
  : srcTracks_(cfg.getParameter<edm::InputTag>("tracks")),
    srcSelectedMuons_(cfg.getParameter<edm::InputTag>("selectedMuons"))
{
  produces<reco::TrackCollection>("tracks");
  produces<reco::PFCandidateCollection>("pfCands");
}

void
ZmumuPFEmbedder::produce(edm::Event& evt, const edm::EventSetup& es)
{
  std::vector<reco::Particle::LorentzVector> selMuonP4s;
   
  edm::Handle<reco::CompositeCandidateCollection> combCandidatesHandle;
  if ( evt.getByLabel(srcSelectedMuons_, combCandidatesHandle) ) {
    if ( combCandidatesHandle->size() >= 1 ) {
      const reco::CompositeCandidate& combCandidate = combCandidatesHandle->at(0); // TF: use only the first combined candidate
      for ( size_t idx = 0; idx < combCandidate.numberOfDaughters(); ++idx ) { 
	selMuonP4s.push_back(combCandidate.daughter(idx)->p4());
      }
    }
  } else {
    edm::Handle<reco::MuonCollection> selectedMuonsHandle;
    if ( evt.getByLabel(srcSelectedMuons_, selectedMuonsHandle) ) {
      for ( size_t idx = 0; idx < selectedMuonsHandle->size(); ++idx ) {
	selMuonP4s.push_back(selectedMuonsHandle->at(idx).p4());
      }
    } else {
      throw cms::Exception("Configuration") 
	<< "Invalid input collection 'selectedMuons' = " << srcSelectedMuons_ << " !!\n";
    }
  }
  
  if ( selMuonP4s.size() <= 2 ) return; // not selected Z --> mu+ mu- event: do nothing
  
  producePFCandColl(evt, &selMuonP4s);
  produceTrackColl(evt, &selMuonP4s);
}

void ZmumuPFEmbedder::producePFCandColl(edm::Event& evt, const std::vector<reco::Particle::LorentzVector>* selMuonP4s)
{
//--- produce collection of PFCandidate excluding muons

  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  evt.getByLabel(srcPFCandidates_, pfCandidates);

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

  evt.put(pfCandidates_woMuons, "pfCands");
}

void ZmumuPFEmbedder::produceTrackColl(edm::Event& evt, const std::vector<reco::Particle::LorentzVector>* selMuonP4s)
{
//--- produce collection of reco::Tracks excluding muons

  edm::Handle<reco::TrackCollection> tracks;
  evt.getByLabel(srcTracks_, tracks);

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

  evt.put(tracks_woMuons, "tracks");
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZmumuPFEmbedder);
