
/** \class ZmumuPFEmbedder
 *
 * Produce collections of PFCandidates and reco::Tracks
 * from which the two muons reconstructed in selected Z --> mu+ mu- events are removed
 * (later to be replaced by simulated tau decay products) 
 * 
 * \author Tomasz Maciej Frueboes
 *
 * \version $Revision: 1.17 $
 *
 * $Id: ZmumuPFEmbedder.cc,v 1.17 2012/12/11 16:29:27 veelken Exp $
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

#include <vector>
#include <algorithm>

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
  template<typename T>
  struct muonMatchInfoType
  {
    muonMatchInfoType(const reco::Particle::LorentzVector& muonP4, const T* pfCandidate_or_track, double dR)
      : muonPt_(muonP4.pt()),
	pfCandidate_or_trackPt_(pfCandidate_or_track->pt()),
	pfCandidate_or_trackCharge_(pfCandidate_or_track->charge()),
	dR_(dR),
	pfCandidate_or_track_(pfCandidate_or_track)
    {}
    ~muonMatchInfoType() {}
    double muonPt_;
    double pfCandidate_or_trackPt_;    
    int pfCandidate_or_trackCharge_;
    double dR_;
    const T* pfCandidate_or_track_;
  };

  template <typename T>
  struct SortMuonMatchInfosDescendingMatchQuality
  {
    bool operator() (const muonMatchInfoType<T>& m1, const muonMatchInfoType<T>& m2)
    {
      // 1st criterion: prefer matches of high Pt
      if ( m1.pfCandidate_or_trackPt_ > (0.5*m1.muonPt_) && m2.pfCandidate_or_trackPt_ < (0.5*m2.muonPt_) ) return true;  // m1 has higher rank than m2
      if ( m1.pfCandidate_or_trackPt_ < (0.5*m1.muonPt_) && m2.pfCandidate_or_trackPt_ > (0.5*m2.muonPt_) ) return false; // m2 has higher rank than m1
      // 2nd criterion: prefer matches to charged particles
      if ( m1.pfCandidate_or_trackCharge_ != 0 && m2.pfCandidate_or_trackCharge_ == 0 ) return true;
      if ( m1.pfCandidate_or_trackCharge_ == 0 && m2.pfCandidate_or_trackCharge_ != 0 ) return false;
      // 3rd criterion: in case multiple matches to high Pt, charged particles exist, 
      //                take particle matched most closely in dR
      return (m1.dR_ < m2.dR_); 
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
  typedef muonMatchInfoType<reco::PFCandidate> muonToPFCandMatchInfoType;
  std::vector<muonToPFCandMatchInfoType> selMuonToPFCandMatches;
  for ( std::vector<reco::Particle::LorentzVector>::const_iterator selMuonP4 = selMuonP4s->begin();
	selMuonP4 != selMuonP4s->end(); ++selMuonP4 ) {
    std::vector<muonToPFCandMatchInfoType> tmpMatches;
    for ( reco::PFCandidateCollection::const_iterator pfCandidate = pfCandidates->begin();
	  pfCandidate != pfCandidates->end(); ++pfCandidate ) {
      double dR = reco::deltaR(pfCandidate->p4(), *selMuonP4);
      if ( dR < dRmatch_ ) tmpMatches.push_back(muonToPFCandMatchInfoType(*selMuonP4, &(*pfCandidate), dR));
    }
    // rank muon-to-pfCandidate matches by quality
    std::sort(tmpMatches.begin(), tmpMatches.end(), SortMuonMatchInfosDescendingMatchQuality<reco::PFCandidate>());
    if ( tmpMatches.size() > 0 ) selMuonToPFCandMatches.push_back(tmpMatches.front());
  }

  for ( reco::PFCandidateCollection::const_iterator pfCandidate = pfCandidates->begin();
	pfCandidate != pfCandidates->end(); ++pfCandidate ) {
    bool isMuon = false;
    for ( std::vector<muonToPFCandMatchInfoType>::const_iterator muonMatchInfo = selMuonToPFCandMatches.begin();
	  muonMatchInfo != selMuonToPFCandMatches.end(); ++muonMatchInfo ) {
      if ( muonMatchInfo->pfCandidate_or_track_ == &(*pfCandidate) ) isMuon = true;
    }
    if ( !isMuon ) pfCandidates_woMuons->push_back(*pfCandidate); // pfCandidate belongs to a selected muon, do not copy
  }

  evt.put(pfCandidates_woMuons, "pfCands");
}

void ZmumuPFEmbedder::produceTrackColl(edm::Event& evt, const std::vector<reco::Particle::LorentzVector>* selMuonP4s)
{
//--- produce collection of reco::Tracks excluding muons

  edm::Handle<reco::TrackCollection> tracks;
  evt.getByLabel(srcTracks_, tracks);

  std::auto_ptr<reco::TrackCollection> tracks_woMuons(new reco::TrackCollection());

  typedef muonMatchInfoType<reco::Track> muonToTrackMatchInfoType;
  std::vector<muonToTrackMatchInfoType> selMuonToTrackMatches;
  for ( std::vector<reco::Particle::LorentzVector>::const_iterator selMuonP4 = selMuonP4s->begin();
	selMuonP4 != selMuonP4s->end(); ++selMuonP4 ) {
    std::vector<muonToTrackMatchInfoType> tmpMatches;
    for ( reco::TrackCollection::const_iterator track = tracks->begin();
	track != tracks->end(); ++track ) {
      double dR = reco::deltaR(track->eta(), track->phi(), selMuonP4->eta(), selMuonP4->phi());
      if ( dR < dRmatch_ ) tmpMatches.push_back(muonToTrackMatchInfoType(*selMuonP4, &(*track), dR));
    }
    // rank muon-to-track matches by quality
    std::sort(tmpMatches.begin(), tmpMatches.end(), SortMuonMatchInfosDescendingMatchQuality<reco::Track>());
    if ( tmpMatches.size() > 0 ) selMuonToTrackMatches.push_back(tmpMatches.front());
  }

  for ( reco::TrackCollection::const_iterator track = tracks->begin();
	track != tracks->end(); ++track ) {
    bool isMuon = false;
    for ( std::vector<muonToTrackMatchInfoType>::const_iterator muonMatchInfo = selMuonToTrackMatches.begin();
	  muonMatchInfo != selMuonToTrackMatches.end(); ++muonMatchInfo ) {
      if ( muonMatchInfo->pfCandidate_or_track_ == &(*track) ) isMuon = true;
    }
    if ( !isMuon ) tracks_woMuons->push_back(*track); // track belongs to a selected muon, do not copy
  }

  evt.put(tracks_woMuons, "tracks");
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZmumuPFEmbedder);
