#ifndef MuonAssociatorByHits_h
#define MuonAssociatorByHits_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimMuon/MCTruth/interface/DTHitAssociator.h"
#include "SimMuon/MCTruth/interface/MuonTruth.h"
#include "SimMuon/MCTruth/interface/RPCHitAssociator.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"


#include <boost/ptr_container/ptr_vector.hpp>

class TrackerTopology;

class MuonAssociatorByHits : public TrackAssociatorBase {
  
 public:
  typedef std::pair <uint32_t, EncodedEventId> SimHitIdpr;
  //typedef std::map<unsigned int, std::vector<SimHitIdpr> > MapOfMatchedIds;
  typedef std::pair<unsigned int,std::vector<SimHitIdpr> > uint_SimHitIdpr_pair;
  typedef boost::ptr_vector<uint_SimHitIdpr_pair> MapOfMatchedIds;
  
  MuonAssociatorByHits( const edm::ParameterSet& );  
  ~MuonAssociatorByHits();
  
  // Get base methods from base class
  using TrackAssociatorBase::associateRecoToSim;
  using TrackAssociatorBase::associateSimToReco;
 
  /* Associate SimTracks to RecoTracks By Hits */
  /// Association Reco To Sim with Collections
  reco::RecoToSimCollection associateRecoToSim(const edm::RefToBaseVector<reco::Track>&,
					       const edm::RefVector<TrackingParticleCollection>&,
					       const edm::Event * event = 0, const edm::EventSetup * setup = 0) const;
  
  /// Association Sim To Reco with Collections
  reco::SimToRecoCollection associateSimToReco(const edm::RefToBaseVector<reco::Track>&,
					       const edm::RefVector<TrackingParticleCollection>&,
					       const edm::Event * event = 0, const edm::EventSetup * setup = 0) const ;

 
  void getMatchedIds
    (MapOfMatchedIds & tracker_matchedIds_valid, MapOfMatchedIds & muon_matchedIds_valid,
     MapOfMatchedIds & tracker_matchedIds_INVALID, MapOfMatchedIds & muon_matchedIds_INVALID,
     int& n_tracker_valid, int& n_dt_valid, int& n_csc_valid, int& n_rpc_valid,
     int& n_tracker_matched_valid, int& n_dt_matched_valid, int& n_csc_matched_valid, int& n_rpc_matched_valid,
     int& n_tracker_INVALID, int& n_dt_INVALID, int& n_csc_INVALID, int& n_rpc_INVALID,
     int& n_tracker_matched_INVALID, int& n_dt_matched_INVALID, int& n_csc_matched_INVALID, int& n_rpc_matched_INVALID,
     trackingRecHit_iterator begin, trackingRecHit_iterator end,
     TrackerHitAssociator* trackertruth, DTHitAssociator& dttruth, MuonTruth& csctruth, RPCHitAssociator& rpctruth, 
     bool printRts, const TrackerTopology *) const;
  
  int getShared(MapOfMatchedIds & matchedIds, TrackingParticleCollection::const_iterator trpart) const;


  enum MuonTrackType { InnerTk, OuterTk, GlobalTk, Segments };
  struct RefToBaseSort { 
    template<typename T> bool operator()(const edm::RefToBase<T> &r1, const edm::RefToBase<T> &r2) const { 
        return (r1.id() == r2.id() ? r1.key() < r2.key() : r1.id() < r2.id()); 
    }
  };
  typedef std::map<edm::RefToBase<reco::Muon>, std::vector<std::pair<TrackingParticleRef, double> >, RefToBaseSort> MuonToSimCollection;
  typedef std::map<TrackingParticleRef, std::vector<std::pair<edm::RefToBase<reco::Muon>, double> > >               SimToMuonCollection;


  void associateMuons(MuonToSimCollection & recoToSim, SimToMuonCollection & simToReco,
                      const edm::RefToBaseVector<reco::Muon> &, MuonTrackType ,
                      const edm::RefVector<TrackingParticleCollection>&,
                      const edm::Event * event = 0, const edm::EventSetup * setup = 0) const ; 

  void associateMuons(MuonToSimCollection & recoToSim, SimToMuonCollection & simToReco,
                      const edm::Handle<edm::View<reco::Muon> > &, MuonTrackType , 
                      const edm::Handle<TrackingParticleCollection>&,
                      const edm::Event * event = 0, const edm::EventSetup * setup = 0) const ;

 private:
  const bool includeZeroHitMuons;    
  const bool acceptOneStubMatchings;
  bool UseTracker;
  bool UseMuon;
  const bool AbsoluteNumberOfHits_track;
  unsigned int NHitCut_track;    
  double EfficiencyCut_track;
  double PurityCut_track;
  const bool AbsoluteNumberOfHits_muon;
  unsigned int NHitCut_muon;    
  double EfficiencyCut_muon;
  double PurityCut_muon;
  const bool UsePixels;
  const bool UseGrouped;
  const bool UseSplitting;
  const bool ThreeHitTracksAreSpecial;
  const bool dumpDT;
  const bool dumpInputCollections;
  const bool crossingframe;
  edm::InputTag simtracksTag;
  edm::InputTag simtracksXFTag;
  const edm::ParameterSet& conf_;

  int LayerFromDetid(const DetId&) const;
  const TrackingRecHit* getHitPtr(edm::OwnVector<TrackingRecHit>::const_iterator iter) const {return &*iter;}
  const TrackingRecHit* getHitPtr(trackingRecHit_iterator iter) const {return &**iter;}

  std::string write_matched_simtracks(const std::vector<SimHitIdpr>&) const;

  /* ==== ALL BELOW THIS IS FOR EXPERTS OR INTERNAL USE ONLY ==== */
  typedef std::vector<std::pair<trackingRecHit_iterator, trackingRecHit_iterator> > TrackHitsCollection;
  struct IndexMatch {
      IndexMatch(size_t index, double global_quality) : idx(index), quality(global_quality) {}
      size_t idx; double quality; 
      bool operator<(const IndexMatch &other) const { return other.quality < quality; } 
  };
  typedef std::map<size_t, std::vector<IndexMatch> > IndexAssociation;
 
  IndexAssociation associateSimToRecoIndices(const TrackHitsCollection &, 
                                             const edm::RefVector<TrackingParticleCollection>&,
					     const edm::Event * event = 0, const edm::EventSetup * setup = 0) const ; 
  IndexAssociation associateRecoToSimIndices(const TrackHitsCollection &, 
                                             const edm::RefVector<TrackingParticleCollection>&,
					     const edm::Event * event = 0, const edm::EventSetup * setup = 0) const ;
  


};

#endif
