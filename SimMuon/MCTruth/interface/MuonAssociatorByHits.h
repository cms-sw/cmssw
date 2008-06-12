#ifndef MuonAssociatorByHits_h
#define MuonAssociatorByHits_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimMuon/MCTruth/interface/DTHitAssociator.h"
#include "SimMuon/MCTruth/interface/MuonTruth.h"
#include "SimMuon/MCTruth/interface/RPCHitAssociator.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

class MuonAssociatorByHits {
  
 public:
  typedef std::pair <uint32_t, EncodedEventId> SimHitIdpr;

  MuonAssociatorByHits( const edm::ParameterSet& );  
  ~MuonAssociatorByHits();
  
  /* Associate SimTracks to RecoTracks By Hits */
 
  /// Association Reco To Sim with Collections
  reco::RecoToSimCollection associateRecoToSim(edm::RefToBaseVector<reco::Track>&,
					       edm::RefVector<TrackingParticleCollection>&,
					       const edm::Event * event = 0, const edm::EventSetup * setup = 0) const;
  
  /// Association Sim To Reco with Collections
  reco::SimToRecoCollection associateSimToReco(edm::RefToBaseVector<reco::Track>&,
					       edm::RefVector<TrackingParticleCollection>&,
					       const edm::Event * event = 0, const edm::EventSetup * setup = 0) const ;
  
  /// compare reco to sim the handle of reco::Track and TrackingParticle collections
  reco::RecoToSimCollection associateRecoToSim(edm::Handle<edm::View<reco::Track> >& tCH, 
					       edm::Handle<TrackingParticleCollection>& tPCH, 
					       const edm::Event * event = 0, const edm::EventSetup * setup = 0) const {
    edm::RefToBaseVector<reco::Track> tc(tCH);
    for (unsigned int j=0; j<tCH->size();j++)
      tc.push_back(edm::RefToBase<reco::Track>(tCH,j));
    
    edm::RefVector<TrackingParticleCollection> tpc(tPCH.id());
    for (unsigned int j=0; j<tPCH->size();j++)
      tpc.push_back(edm::Ref<TrackingParticleCollection>(tPCH,j));
    
    return associateRecoToSim(tc,tpc,event,setup);
  }
  
  /// compare reco to sim the handle of reco::Track and TrackingParticle collections
  reco::SimToRecoCollection associateSimToReco(edm::Handle<edm::View<reco::Track> >& tCH, 
					       edm::Handle<TrackingParticleCollection>& tPCH,
					       const edm::Event * event = 0, const edm::EventSetup * setup = 0) const {
    edm::RefToBaseVector<reco::Track> tc(tCH);
    for (unsigned int j=0; j<tCH->size();j++)
      tc.push_back(edm::RefToBase<reco::Track>(tCH,j));
    
    edm::RefVector<TrackingParticleCollection> tpc(tPCH.id());
    for (unsigned int j=0; j<tPCH->size();j++)
      tpc.push_back(edm::Ref<TrackingParticleCollection>(tPCH,j));
    
    return associateSimToReco(tc,tpc,event,setup);
  }  
  
  template<typename iter>
    void getMatchedIds(std::vector<SimHitIdpr>& tracker_matchedIds, std::vector<SimHitIdpr>& muon_matchedIds, 
		       int& n_valid_hits,int& n_tracker_valid_hits,int& n_dt_valid_hits,int& n_csc_valid_hits,int& n_rpc_valid_hits,
		       int& n_matched_hits,int& n_tracker_matched_hits,int& n_dt_matched_hits,int& n_csc_matched_hits,int& n_rpc_matched_hits,
		       iter begin, iter end,
		       TrackerHitAssociator* trackertruth, 
		       DTHitAssociator& dttruth, MuonTruth& csctruth, RPCHitAssociator& rpctruth, bool printRts) const;
  
  int getShared(std::vector<SimHitIdpr>& matchedIds, std::vector<SimHitIdpr>& idcachev,
		TrackingParticleCollection::const_iterator trpart) const;
  
 private:
  const bool AbsoluteNumberOfHits_track;
  const unsigned int MinHitCut_track;    
  const bool AbsoluteNumberOfHits_muon;
  const unsigned int MinHitCut_muon;    
  const double PurityCut_track;
  const double PurityCut_muon;
  const bool SimToReco_useTracker;
  const double EfficiencyCut_track;
  const bool SimToReco_useMuon;
  const double EfficiencyCut_muon;
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

};

#endif
