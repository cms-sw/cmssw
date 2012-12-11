
/** \class ZmumuPFEmbedder
 *
 * Produce collections of PFCandidates and reco::Tracks
 * from which the two muons reconstructed in selected Z --> mu+ mu- events are removed
 * (later to be replaced by simulated tau decay products) 
 * 
 * \author Tomasz Maciej Frueboes
 *
 * \version $Revision: 1.16 $
 *
 * $Id: ZmumuPFEmbedder.cc,v 1.16 2012/11/25 15:43:13 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

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

  double dRmatch_;
};

ZmumuPFEmbedder::ZmumuPFEmbedder(const edm::ParameterSet& cfg)
  : srcTracks_(cfg.getParameter<edm::InputTag>("tracks")),
    srcPFCandidates_(cfg.getParameter<edm::InputTag>("pfCands")),
    srcSelectedMuons_(cfg.getParameter<edm::InputTag>("selectedMuons")),
    dRmatch_(cfg.getParameter<double>("dRmatch"))
{
  produces<reco::TrackCollection>("tracks");
  produces<reco::PFCandidateCollection>("pfCands");
}

namespace
{
  template <typename T>
  struct higherPtT
  {
    bool operator() (const T& t1, const T& t2)
    {
      return (t1.pt() > t2.pt());
    }
  };
}

void
ZmumuPFEmbedder::produce(edm::Event& evt, const edm::EventSetup& es)
{
  std::vector<reco::CandidateBaseRef> selMuons = getSelMuons(evt, srcSelectedMuons_);
  const reco::CandidateBaseRef muPlus  = getTheMuPlus(selMuons);
  const reco::CandidateBaseRef muMinus = getTheMuMinus(selMuons);

  std::vector<reco::Particle::LorentzVector> selMuonP4s;
  if ( muPlus.isNonnull()  ) selMuonP4s.push_back(muPlus->p4());
  if ( muMinus.isNonnull() ) selMuonP4s.push_back(muMinus->p4());

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

    if ( dRmin < dRmatch_ ) continue; // it is a selected muon, do not copy
       
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
    bool isMuonTrack = (dRmin < dRmatch_);
    if ( !isMuonTrack ) { // track belongs to a selected muon, do not copy
      tracks_woMuons->push_back(*track);
    }
  }
	
  evt.put(tracks_woMuons, "tracks");
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZmumuPFEmbedder);
