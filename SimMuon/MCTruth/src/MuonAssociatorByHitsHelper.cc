#include "SimMuon/MCTruth/interface/MuonAssociatorByHitsHelper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include <sstream>

using namespace reco;
using namespace std;

MuonAssociatorByHitsHelper::MuonAssociatorByHitsHelper (const edm::ParameterSet& conf) :  
  includeZeroHitMuons(conf.getParameter<bool>("includeZeroHitMuons")),
  acceptOneStubMatchings(conf.getParameter<bool>("acceptOneStubMatchings")),
  UseTracker(conf.getParameter<bool>("UseTracker")),
  UseMuon(conf.getParameter<bool>("UseMuon")),
  AbsoluteNumberOfHits_track(conf.getParameter<bool>("AbsoluteNumberOfHits_track")),
  NHitCut_track(conf.getParameter<unsigned int>("NHitCut_track")),
  EfficiencyCut_track(conf.getParameter<double>("EfficiencyCut_track")),
  PurityCut_track(conf.getParameter<double>("PurityCut_track")),
  AbsoluteNumberOfHits_muon(conf.getParameter<bool>("AbsoluteNumberOfHits_muon")),
  NHitCut_muon(conf.getParameter<unsigned int>("NHitCut_muon")),
  EfficiencyCut_muon(conf.getParameter<double>("EfficiencyCut_muon")),
  PurityCut_muon(conf.getParameter<double>("PurityCut_muon")),
  UsePixels(conf.getParameter<bool>("UsePixels")),
  UseGrouped(conf.getParameter<bool>("UseGrouped")),
  UseSplitting(conf.getParameter<bool>("UseSplitting")),
  ThreeHitTracksAreSpecial(conf.getParameter<bool>("ThreeHitTracksAreSpecial")),
  dumpDT(conf.getParameter<bool>("dumpDT"))
{
  edm::LogVerbatim("MuonAssociatorByHitsHelper") << "constructing  MuonAssociatorByHitsHelper" << conf.dump();

  // up to the user in the other cases - print a message
  if (UseTracker) edm::LogVerbatim("MuonAssociatorByHitsHelper")<<"\n UseTracker = TRUE  : Tracker SimHits and RecHits WILL be counted";
  else edm::LogVerbatim("MuonAssociatorByHitsHelper") <<"\n UseTracker = FALSE : Tracker SimHits and RecHits WILL NOT be counted";
  
  // up to the user in the other cases - print a message
  if (UseMuon) edm::LogVerbatim("MuonAssociatorByHitsHelper")<<" UseMuon = TRUE  : Muon SimHits and RecHits WILL be counted";
  else edm::LogVerbatim("MuonAssociatorByHitsHelper") <<" UseMuon = FALSE : Muon SimHits and RecHits WILL NOT be counted"<<endl;
  
  // check consistency of the configuration when allowing zero-hit muon matching (counting invalid hits)
  if (includeZeroHitMuons) {
    edm::LogVerbatim("MuonAssociatorByHitsHelper") 
      <<"\n includeZeroHitMuons = TRUE"
      <<"\n ==> (re)set NHitCut_muon = 0, PurityCut_muon = 0, EfficiencyCut_muon = 0"<<endl;
    NHitCut_muon = 0;
    PurityCut_muon = 0.;
    EfficiencyCut_muon = 0.;
  }

}

MuonAssociatorByHitsHelper::IndexAssociation
MuonAssociatorByHitsHelper::associateRecoToSimIndices(const TrackHitsCollection & tC,
                                                      const edm::RefVector<TrackingParticleCollection>& TPCollectionH,
                                                      const Resources& resources) const {

  auto tTopo = resources.tTopo_;
  auto trackertruth = resources.trackerHitAssoc_;
  auto const & csctruth = *resources.cscHitAssoc_;
  auto const& dttruth = *resources.dtHitAssoc_;
  auto const& rpctruth = *resources.rpcHitAssoc_;

  int tracker_nshared = 0;
  int muon_nshared = 0;
  int global_nshared = 0;

  double tracker_quality = 0;
  double tracker_quality_cut;
  if (AbsoluteNumberOfHits_track) tracker_quality_cut = static_cast<double>(NHitCut_track); 
  else tracker_quality_cut = PurityCut_track;

  double muon_quality = 0;
  double muon_quality_cut;
  if (AbsoluteNumberOfHits_muon) muon_quality_cut = static_cast<double>(NHitCut_muon); 
  else muon_quality_cut = PurityCut_muon;

  double global_quality = 0;
  
  MapOfMatchedIds tracker_matchedIds_valid, muon_matchedIds_valid;
  MapOfMatchedIds tracker_matchedIds_INVALID, muon_matchedIds_INVALID;

  IndexAssociation     outputCollection;
  bool printRtS(true);

  TrackingParticleCollection tPC;
  tPC.reserve(TPCollectionH.size());
  for(auto const& ref: TPCollectionH) {
    tPC.push_back(*ref);
  }

  if(resources.diagnostics_) {
    resources.diagnostics_(tC,tPC);
  }

  int tindex=0;
  for (TrackHitsCollection::const_iterator track=tC.begin(); track!=tC.end(); track++, tindex++) {
    edm::LogVerbatim("MuonAssociatorByHitsHelper")
      <<"\n"<<"reco::Track "<<tindex
      <<", number of RecHits = "<< (track->second - track->first) << "\n";
    tracker_matchedIds_valid.clear();
    muon_matchedIds_valid.clear();

    tracker_matchedIds_INVALID.clear();
    muon_matchedIds_INVALID.clear();

    bool this_track_matched = false;
    int n_matching_simhits = 0;

    // all hits = valid +INVALID
    int n_all         = 0;        
    int n_tracker_all = 0;
    int n_dt_all      = 0;     
    int n_csc_all     = 0;    
    int n_rpc_all     = 0;    

    int n_valid         = 0;        
    int n_tracker_valid = 0;
    int n_muon_valid    = 0;   
    int n_dt_valid      = 0;     
    int n_csc_valid     = 0;    
    int n_rpc_valid     = 0;    

    int n_tracker_matched_valid = 0;
    int n_muon_matched_valid    = 0;   
    int n_dt_matched_valid      = 0;     
    int n_csc_matched_valid     = 0;    
    int n_rpc_matched_valid     = 0;    

    int n_INVALID         = 0;        
    int n_tracker_INVALID = 0;
    int n_muon_INVALID    = 0;   
    int n_dt_INVALID      = 0;     
    int n_csc_INVALID     = 0;    
    int n_rpc_INVALID     = 0;    
    
    int n_tracker_matched_INVALID = 0;
    int n_muon_matched_INVALID    = 0;     
    int n_dt_matched_INVALID      = 0;     
    int n_csc_matched_INVALID     = 0;    
    int n_rpc_matched_INVALID     = 0;    
    
    printRtS = true;
    getMatchedIds(tracker_matchedIds_valid, muon_matchedIds_valid,
		  tracker_matchedIds_INVALID, muon_matchedIds_INVALID,       
		  n_tracker_valid, n_dt_valid, n_csc_valid, n_rpc_valid,
		  n_tracker_matched_valid, n_dt_matched_valid, n_csc_matched_valid, n_rpc_matched_valid,
		  n_tracker_INVALID, n_dt_INVALID, n_csc_INVALID, n_rpc_INVALID,
		  n_tracker_matched_INVALID, n_dt_matched_INVALID, n_csc_matched_INVALID, n_rpc_matched_INVALID,
                  track->first, track->second,
		  trackertruth, dttruth, csctruth, rpctruth,
		  printRtS,tTopo);
    
    n_matching_simhits = tracker_matchedIds_valid.size() + muon_matchedIds_valid.size() + 
                         tracker_matchedIds_INVALID.size() +muon_matchedIds_INVALID.size(); 

    n_muon_valid   = n_dt_valid + n_csc_valid + n_rpc_valid;
    n_valid        = n_tracker_valid + n_muon_valid;
    n_muon_INVALID = n_dt_INVALID + n_csc_INVALID + n_rpc_INVALID;
    n_INVALID      = n_tracker_INVALID + n_muon_INVALID;

    // all used hits (valid+INVALID), defined by UseTracker, UseMuon
    n_tracker_all = n_tracker_valid + n_tracker_INVALID;
    n_dt_all      = n_dt_valid  + n_dt_INVALID;
    n_csc_all     = n_csc_valid + n_csc_INVALID;
    n_rpc_all     = n_rpc_valid + n_rpc_INVALID;
    n_all         = n_valid + n_INVALID;

    n_muon_matched_valid   = n_dt_matched_valid + n_csc_matched_valid + n_rpc_matched_valid;
    n_muon_matched_INVALID = n_dt_matched_INVALID + n_csc_matched_INVALID + n_rpc_matched_INVALID;

    // selected hits are set initially to valid hits
    int n_tracker_selected_hits = n_tracker_valid;
    int n_muon_selected_hits    = n_muon_valid;
    int n_dt_selected_hits      = n_dt_valid;
    int n_csc_selected_hits     = n_csc_valid;
    int n_rpc_selected_hits     = n_rpc_valid;

    // matched hits are a subsample of the selected hits
    int n_tracker_matched = n_tracker_matched_valid;
    int n_muon_matched    = n_muon_matched_valid;
    int n_dt_matched      = n_dt_matched_valid;
    int n_csc_matched     = n_csc_matched_valid;
    int n_rpc_matched     = n_rpc_matched_valid;

    std::string InvMuonHits, ZeroHitMuon;
    
    if (includeZeroHitMuons && n_muon_valid==0 && n_muon_INVALID!=0) {
      // selected muon hits = INVALID when (useZeroHitMuons == True) and track has no valid muon hits

      InvMuonHits = " ***INVALID MUON HITS***";
      ZeroHitMuon = " ***ZERO-HIT MUON***";

      n_muon_selected_hits = n_muon_INVALID;
      n_dt_selected_hits   = n_dt_INVALID;
      n_csc_selected_hits  = n_csc_INVALID;
      n_rpc_selected_hits  = n_rpc_INVALID;

      n_muon_matched = n_muon_matched_INVALID;
      n_dt_matched   = n_dt_matched_INVALID;
      n_csc_matched  = n_csc_matched_INVALID;
      n_rpc_matched  = n_rpc_matched_INVALID;      
    }

    int n_selected_hits = n_tracker_selected_hits + n_muon_selected_hits;
    int n_matched = n_tracker_matched + n_muon_matched;

    edm::LogVerbatim("MuonAssociatorByHitsHelper")
      <<"\n"<<"# TrackingRecHits: "<<(track->second - track->first) 
      <<"\n"<< "# used RecHits     = " << n_all <<" ("<<n_tracker_all<<"/"
      <<n_dt_all<<"/"<<n_csc_all<<"/"<<n_rpc_all<<" in Tracker/DT/CSC/RPC)"<<", obtained from " << n_matching_simhits << " SimHits"
      <<"\n"<< "# selected RecHits = " <<n_selected_hits <<" (" <<n_tracker_selected_hits<<"/"
      <<n_dt_selected_hits<<"/"<<n_csc_selected_hits<<"/"<<n_rpc_selected_hits<<" in Tracker/DT/CSC/RPC)"<<InvMuonHits
      <<"\n"<< "# matched RecHits  = " <<n_matched<<" ("<<n_tracker_matched<<"/"
      <<n_dt_matched<<"/"<<n_csc_matched<<"/"<<n_rpc_matched<<" in Tracker/DT/CSC/RPC)";

    if (n_all>0 && n_matching_simhits == 0)
      edm::LogWarning("MuonAssociatorByHitsHelper")
	<<"*** WARNING in MuonAssociatorByHitsHelper::associateRecoToSim: no matching PSimHit found for this reco::Track !";

    if (n_matching_simhits != 0) {
      edm::LogVerbatim("MuonAssociatorByHitsHelper")
	<<"\n"<< "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
	<<"\n"<< "reco::Track "<<tindex<<ZeroHitMuon
	<<"\n\t"<< "made of "<<n_selected_hits<<" selected RecHits (tracker:"<<n_tracker_selected_hits<<"/muons:"<<n_muon_selected_hits<<")";

      int tpindex = 0;
      for (TrackingParticleCollection::const_iterator trpart = tPC.begin(); trpart != tPC.end(); ++trpart, ++tpindex) {
	tracker_nshared = getShared(tracker_matchedIds_valid, trpart);
	muon_nshared = getShared(muon_matchedIds_valid, trpart);

	if (includeZeroHitMuons && n_muon_valid==0 && n_muon_INVALID!=0) 
	  muon_nshared = getShared(muon_matchedIds_INVALID, trpart);

        global_nshared = tracker_nshared + muon_nshared;

	if (AbsoluteNumberOfHits_track) tracker_quality = static_cast<double>(tracker_nshared);
	else if(n_tracker_selected_hits != 0) tracker_quality = (static_cast<double>(tracker_nshared)/static_cast<double>(n_tracker_selected_hits));
	else tracker_quality = 0;

	if (AbsoluteNumberOfHits_muon) muon_quality = static_cast<double>(muon_nshared);
	else if(n_muon_selected_hits != 0) muon_quality = (static_cast<double>(muon_nshared)/static_cast<double>(n_muon_selected_hits));
	else muon_quality = 0;

	// global_quality used to order the matching TPs
	if (n_selected_hits != 0) {
            if (AbsoluteNumberOfHits_muon && AbsoluteNumberOfHits_track) 
                global_quality = global_nshared;
            else
                global_quality = (static_cast<double>(global_nshared)/static_cast<double>(n_selected_hits));
        } else global_quality = 0;

	bool trackerOk = false;
	if (n_tracker_selected_hits != 0) {
	  if (tracker_quality > tracker_quality_cut) trackerOk = true;
	  //if a track has just 3 hits in the Tracker we require that all 3 hits are shared
	  if (ThreeHitTracksAreSpecial && n_tracker_selected_hits==3 && tracker_nshared<3) trackerOk = false;
	}
	
	bool muonOk = false;
	if (n_muon_selected_hits != 0) {
	  if (muon_quality > muon_quality_cut) muonOk = true;
	}
	
	// (matchOk) has to account for different track types (tracker-only, standalone muons, global muons)
	bool matchOk = trackerOk || muonOk;

	// only for global muons: match both tracker and muon stub unless (acceptOneStubMatchings==true)
	if (!acceptOneStubMatchings && n_tracker_selected_hits!=0 && n_muon_selected_hits!=0)
	  matchOk = trackerOk && muonOk;
	
	if (matchOk) {

          outputCollection[tindex].push_back(IndexMatch(tpindex, global_quality));
	  this_track_matched = true;

	  edm::LogVerbatim("MuonAssociatorByHitsHelper")
	    << "\n\t"<<" **MATCHED** with quality = "<<global_quality<<" (tracker: "<<tracker_quality<<" / muon: "<<muon_quality<<")"
	    << "\n\t"<<"   N shared hits = "<<global_nshared<<" (tracker: "<<tracker_nshared<<" / muon: "<<muon_nshared<<")"
	    <<"\n"<< "   to: TrackingParticle " <<tpindex<<", q = "<<(*trpart).charge()<<", p = "<<(*trpart).p()
	    <<", pT = "<<(*trpart).pt()<<", eta = "<<(*trpart).eta()<<", phi = "<<(*trpart).phi()
	    <<"\n\t"<< " pdg code = "<<(*trpart).pdgId()<<", made of "<<(*trpart).numberOfHits()<<" PSimHits"
	    <<" from "<<(*trpart).g4Tracks().size()<<" SimTrack:";
	  for(TrackingParticle::g4t_iterator g4T=(*trpart).g4Track_begin(); 
	      g4T!=(*trpart).g4Track_end(); 
	      ++g4T) {
	    edm::LogVerbatim("MuonAssociatorByHitsHelper")
	      <<"\t"<< " Id:"<<(*g4T).trackId()<<"/Evt:("<<(*g4T).eventId().event()<<","<<(*g4T).eventId().bunchCrossing()<<")";	    
	  }
	}
	else {
	  // print something only if this TrackingParticle shares some hits with the current reco::Track
	  if (global_nshared != 0) 
	    edm::LogVerbatim("MuonAssociatorByHitsHelper")
	      <<"\n\t"<<" NOT matched to TrackingParticle "<<tpindex
	      << " with quality = "<<global_quality<<" (tracker: "<<tracker_quality<<" / muon: "<<muon_quality<<")"
	      << "\n"<< "   N shared hits = "<<global_nshared<<" (tracker: "<<tracker_nshared<<" / muon: "<<muon_nshared<<")";
	}
	
      }    //  loop over TrackingParticle

      if (!this_track_matched) {
	edm::LogVerbatim("MuonAssociatorByHitsHelper")
	  <<"\n"<<" NOT matched to any TrackingParticle";
      }
      
      edm::LogVerbatim("MuonAssociatorByHitsHelper")
	<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<"\n";
     
    }    //  if(n_matching_simhits != 0)
    
  }    // loop over reco::Track

  if (!tC.size()) 
    edm::LogVerbatim("MuonAssociatorByHitsHelper")<<"0 reconstructed tracks (-->> 0 associated !)";

  for (IndexAssociation::iterator it = outputCollection.begin(), ed = outputCollection.end(); it != ed; ++it) {
    std::sort(it->second.begin(), it->second.end());
  }
  return outputCollection;
}


MuonAssociatorByHitsHelper::IndexAssociation
MuonAssociatorByHitsHelper::associateSimToRecoIndices( const TrackHitsCollection & tC, 
                                                       const edm::RefVector<TrackingParticleCollection>& TPCollectionH,
                                                       const Resources& resources) const {
  auto tTopo = resources.tTopo_;
  auto trackertruth = resources.trackerHitAssoc_;
  auto const & csctruth = *resources.cscHitAssoc_;
  auto const& dttruth = *resources.dtHitAssoc_;
  auto const& rpctruth = *resources.rpcHitAssoc_;

  int tracker_nshared = 0;
  int muon_nshared = 0;
  int global_nshared = 0;

  double tracker_quality = 0;
  double tracker_quality_cut;
  if (AbsoluteNumberOfHits_track) tracker_quality_cut = static_cast<double>(NHitCut_track); 
  else tracker_quality_cut = EfficiencyCut_track;
  
  double muon_quality = 0;
  double muon_quality_cut;
  if (AbsoluteNumberOfHits_muon) muon_quality_cut = static_cast<double>(NHitCut_muon); 
  else muon_quality_cut = EfficiencyCut_muon;

  double global_quality = 0;

  double tracker_purity = 0;
  double muon_purity = 0;
  double global_purity = 0;
  
  MapOfMatchedIds tracker_matchedIds_valid, muon_matchedIds_valid;
  MapOfMatchedIds tracker_matchedIds_INVALID, muon_matchedIds_INVALID;

  IndexAssociation  outputCollection;


  bool printRtS(true);

  TrackingParticleCollection tPC;
  tPC.reserve(TPCollectionH.size());
  for(auto const& ref: TPCollectionH) {
    tPC.push_back(*ref);
  }

  bool any_trackingParticle_matched = false;
  
  int tindex=0;
  for (TrackHitsCollection::const_iterator track=tC.begin(); track!=tC.end(); track++, tindex++) {
    if (printRtS) edm::LogVerbatim("MuonAssociatorByHitsHelper")
      <<"\n"<<"reco::Track "<<tindex
      <<", number of RecHits = "<< (track->second - track->first) << "\n";
    
    tracker_matchedIds_valid.clear();
    muon_matchedIds_valid.clear();

    tracker_matchedIds_INVALID.clear();
    muon_matchedIds_INVALID.clear();

    int n_matching_simhits = 0;

    // all hits = valid +INVALID
    int n_all         = 0;        
    int n_tracker_all = 0;
    int n_dt_all      = 0;     
    int n_csc_all     = 0;    
    int n_rpc_all     = 0;    

    int n_valid         = 0;        
    int n_tracker_valid = 0;
    int n_muon_valid    = 0;   
    int n_dt_valid      = 0;     
    int n_csc_valid     = 0;    
    int n_rpc_valid     = 0;    

    int n_tracker_matched_valid = 0;
    int n_muon_matched_valid    = 0;   
    int n_dt_matched_valid      = 0;     
    int n_csc_matched_valid     = 0;    
    int n_rpc_matched_valid     = 0;    

    int n_INVALID         = 0;        
    int n_tracker_INVALID = 0;
    int n_muon_INVALID    = 0;   
    int n_dt_INVALID      = 0;     
    int n_csc_INVALID     = 0;    
    int n_rpc_INVALID     = 0;    
    
    int n_tracker_matched_INVALID = 0;
    int n_muon_matched_INVALID    = 0;     
    int n_dt_matched_INVALID      = 0;     
    int n_csc_matched_INVALID     = 0;    
    int n_rpc_matched_INVALID     = 0;    
    
    printRtS = false;
    getMatchedIds(tracker_matchedIds_valid, muon_matchedIds_valid,
		  tracker_matchedIds_INVALID, muon_matchedIds_INVALID,       
		  n_tracker_valid, n_dt_valid, n_csc_valid, n_rpc_valid,
		  n_tracker_matched_valid, n_dt_matched_valid, n_csc_matched_valid, n_rpc_matched_valid,
		  n_tracker_INVALID, n_dt_INVALID, n_csc_INVALID, n_rpc_INVALID,
		  n_tracker_matched_INVALID, n_dt_matched_INVALID, n_csc_matched_INVALID, n_rpc_matched_INVALID,
                  track->first, track->second,
		  trackertruth, dttruth, csctruth, rpctruth,
		  printRtS,tTopo);
    
    n_matching_simhits = tracker_matchedIds_valid.size() + muon_matchedIds_valid.size() + 
                         tracker_matchedIds_INVALID.size() +muon_matchedIds_INVALID.size(); 

    n_muon_valid   = n_dt_valid + n_csc_valid + n_rpc_valid;
    n_valid        = n_tracker_valid + n_muon_valid;
    n_muon_INVALID = n_dt_INVALID + n_csc_INVALID + n_rpc_INVALID;
    n_INVALID      = n_tracker_INVALID + n_muon_INVALID;

    // all used hits (valid+INVALID), defined by UseTracker, UseMuon
    n_tracker_all = n_tracker_valid + n_tracker_INVALID;
    n_dt_all      = n_dt_valid  + n_dt_INVALID;
    n_csc_all     = n_csc_valid + n_csc_INVALID;
    n_rpc_all     = n_rpc_valid + n_rpc_INVALID;
    n_all         = n_valid + n_INVALID;

    n_muon_matched_valid   = n_dt_matched_valid + n_csc_matched_valid + n_rpc_matched_valid;
    n_muon_matched_INVALID = n_dt_matched_INVALID + n_csc_matched_INVALID + n_rpc_matched_INVALID;

     // selected hits are set initially to valid hits
    int n_tracker_selected_hits = n_tracker_valid;
    int n_muon_selected_hits    = n_muon_valid;
    int n_dt_selected_hits      = n_dt_valid;
    int n_csc_selected_hits     = n_csc_valid;
    int n_rpc_selected_hits     = n_rpc_valid;

    // matched hits are a subsample of the selected hits
    int n_tracker_matched = n_tracker_matched_valid;
    int n_muon_matched    = n_muon_matched_valid;
    int n_dt_matched      = n_dt_matched_valid;
    int n_csc_matched     = n_csc_matched_valid;
    int n_rpc_matched     = n_rpc_matched_valid;

    std::string InvMuonHits, ZeroHitMuon;

    if (includeZeroHitMuons && n_muon_valid==0 && n_muon_INVALID!=0) {
      // selected muon hits = INVALID when (useZeroHitMuons == True) and track has no valid muon hits
      
      InvMuonHits = " ***INVALID MUON HITS***";
      ZeroHitMuon = " ***ZERO-HIT MUON***";

      n_muon_selected_hits = n_muon_INVALID;
      n_dt_selected_hits   = n_dt_INVALID;
      n_csc_selected_hits  = n_csc_INVALID;
      n_rpc_selected_hits  = n_rpc_INVALID;

      n_muon_matched = n_muon_matched_INVALID;
      n_dt_matched   = n_dt_matched_INVALID;
      n_csc_matched  = n_csc_matched_INVALID;
      n_rpc_matched  = n_rpc_matched_INVALID;
    }

    int n_selected_hits = n_tracker_selected_hits + n_muon_selected_hits;
    int n_matched = n_tracker_matched + n_muon_matched;

    if (printRtS) edm::LogVerbatim("MuonAssociatorByHitsHelper")
      <<"\n"<<"# TrackingRecHits: "<<(track->second - track->first) 
      <<"\n"<< "# used RecHits     = " <<n_all    <<" ("<<n_tracker_all<<"/"
      <<n_dt_all<<"/"<<n_csc_all<<"/"<<n_rpc_all<<" in Tracker/DT/CSC/RPC)"<<", obtained from " << n_matching_simhits << " SimHits"
      <<"\n"<< "# selected RecHits = " <<n_selected_hits  <<" (" <<n_tracker_selected_hits<<"/"
      <<n_dt_selected_hits<<"/"<<n_csc_selected_hits<<"/"<<n_rpc_selected_hits<<" in Tracker/DT/CSC/RPC)"<<InvMuonHits
      <<"\n"<< "# matched RecHits = " <<n_matched<<" ("<<n_tracker_matched<<"/"
      <<n_dt_matched<<"/"<<n_csc_matched<<"/"<<n_rpc_matched<<" in Tracker/DT/CSC/RPC)";
    
    if (printRtS && n_all>0 && n_matching_simhits==0)
      edm::LogWarning("MuonAssociatorByHitsHelper")
	<<"*** WARNING in MuonAssociatorByHitsHelper::associateSimToReco: no matching PSimHit found for this reco::Track !";
    
    if (n_matching_simhits != 0) {
      int tpindex =0;
      for (TrackingParticleCollection::const_iterator trpart = tPC.begin(); trpart != tPC.end(); ++trpart, ++tpindex) {

	//	int n_tracker_simhits = 0;
	int n_tracker_recounted_simhits = 0; 
	int n_muon_simhits = 0; 
	int n_global_simhits = 0; 
	//	std::vector<PSimHit> tphits;

	int n_tracker_selected_simhits = 0;
	int n_muon_selected_simhits = 0; 
	int n_global_selected_simhits = 0; 

	// shared hits are counted over the selected ones
	tracker_nshared = getShared(tracker_matchedIds_valid, trpart);
	muon_nshared = getShared(muon_matchedIds_valid, trpart);

        if (includeZeroHitMuons && n_muon_valid==0 && n_muon_INVALID!=0)
	  muon_nshared = getShared(muon_matchedIds_INVALID, trpart);
	
        global_nshared = tracker_nshared + muon_nshared;	
        if (global_nshared == 0) continue; // if this TP shares no hits with the current reco::Track loop over 

	// This does not work with the new TP interface 
	/*
	for(std::vector<PSimHit>::const_iterator TPhit = trpart->pSimHit_begin(); TPhit != trpart->pSimHit_end(); TPhit++) {
          DetId dId = DetId(TPhit->detUnitId());
	  DetId::Detector detector = dId.det();
	  
	  if (detector == DetId::Tracker) {
	    n_tracker_simhits++;

	    unsigned int subdetId = static_cast<unsigned int>(dId.subdetId());
	    if (!UsePixels && (subdetId==PixelSubdetector::PixelBarrel || subdetId==PixelSubdetector::PixelEndcap) )
	      continue;

	    SiStripDetId* stripDetId = 0;
	    if (subdetId==SiStripDetId::TIB||subdetId==SiStripDetId::TOB||
		subdetId==SiStripDetId::TID||subdetId==SiStripDetId::TEC)
	      stripDetId= new SiStripDetId(dId);
	    
	    bool newhit = true;
	    for(std::vector<PSimHit>::const_iterator TPhitOK = tphits.begin(); TPhitOK != tphits.end(); TPhitOK++) {
	      DetId dIdOK = DetId(TPhitOK->detUnitId());
	      //no grouped, no splitting
	      if (!UseGrouped && !UseSplitting)
		if (tTopo->layer(dId)==tTopo->layer(dIdOK) &&
		    dId.subdetId()==dIdOK.subdetId()) newhit = false;
	      //no grouped, splitting
	      if (!UseGrouped && UseSplitting)
		if (tTopo->layer(dId)==tTopo->layer(dIdOK) &&
		    dId.subdetId()==dIdOK.subdetId() &&
		    (stripDetId==0 || stripDetId->partnerDetId()!=dIdOK.rawId()))
		  newhit = false;
	      //grouped, no splitting
	      if (UseGrouped && !UseSplitting)
		if ( tTopo->layer(dId)== tTopo->layer(dIdOK) &&
		    dId.subdetId()==dIdOK.subdetId() &&
		    stripDetId!=0 && stripDetId->partnerDetId()==dIdOK.rawId())
		  newhit = false;
	      //grouped, splitting
	      if (UseGrouped && UseSplitting)
		newhit = true;
	    }
	    if (newhit) {
	      tphits.push_back(*TPhit);
	    }
	    delete stripDetId;
	  }
	  else if (detector == DetId::Muon) {
	    n_muon_simhits++;
	    
	    // discard BAD CSC chambers (ME4/2) from hit counting
	    if (dId.subdetId() == MuonSubdetId::CSC) {
              if (csctruth.cscBadChambers->isInBadChamber(CSCDetId(dId))) {
		// edm::LogVerbatim("MuonAssociatorByHitsHelper")<<"This PSimHit is in a BAD CSC chamber, CSCDetId = "<<CSCDetId(dId);
		n_muon_simhits--;
	      }
	    }
	    
	  }
	}
	*/
	//	n_tracker_recounted_simhits = tphits.size();

        // adapt to new TP interface: this gives the total number of hits in tracker
        //   should reproduce the behaviour of UseGrouped=UseSplitting=.true.
	n_tracker_recounted_simhits = trpart->numberOfTrackerHits();
        //   numberOfHits() gives the total number of hits (tracker + muons)
        n_muon_simhits = trpart->numberOfHits() - trpart->numberOfTrackerHits();

        // Handle the case of TrackingParticles that don't have PSimHits inside, e.g. because they were made on RECOSIM only.
        if (trpart->numberOfHits()==0) {
            // FIXME this can be made better, counting the digiSimLinks associated to this TP, but perhaps it's not worth it
            n_tracker_recounted_simhits = tracker_nshared;
            n_muon_simhits = muon_nshared;
        }	
	n_global_simhits = n_tracker_recounted_simhits + n_muon_simhits;

	if (UseMuon) {
	  n_muon_selected_simhits = n_muon_simhits;
	  n_global_selected_simhits = n_muon_selected_simhits;
	}
	if (UseTracker) {
	  n_tracker_selected_simhits = n_tracker_recounted_simhits;
	  n_global_selected_simhits += n_tracker_selected_simhits;
	}

	if (AbsoluteNumberOfHits_track) tracker_quality = static_cast<double>(tracker_nshared);
	else if (n_tracker_selected_simhits!=0) 
	  tracker_quality = static_cast<double>(tracker_nshared)/static_cast<double>(n_tracker_selected_simhits);
	else tracker_quality = 0;
	
	if (AbsoluteNumberOfHits_muon) muon_quality = static_cast<double>(muon_nshared);
	else if (n_muon_selected_simhits!=0) 
	  muon_quality = static_cast<double>(muon_nshared)/static_cast<double>(n_muon_selected_simhits);
	else muon_quality = 0;

	// global_quality used to order the matching tracks
	if (n_global_selected_simhits != 0) {
	  if (AbsoluteNumberOfHits_muon && AbsoluteNumberOfHits_track)
	    global_quality = global_nshared;
	  else 
	    global_quality = static_cast<double>(global_nshared)/static_cast<double>(n_global_selected_simhits);
	} 
	else global_quality = 0;

	// global purity
	if (n_selected_hits != 0) {
	  if (AbsoluteNumberOfHits_muon && AbsoluteNumberOfHits_track)
	    global_purity = global_nshared;
	  else
	    global_purity = static_cast<double>(global_nshared)/static_cast<double>(n_selected_hits);
	}
	else global_purity = 0;
	
	bool trackerOk = false;
	if (n_tracker_selected_hits != 0) {
	  if (tracker_quality > tracker_quality_cut) trackerOk = true;
	  
	  tracker_purity = static_cast<double>(tracker_nshared)/static_cast<double>(n_tracker_selected_hits);
	  if (AbsoluteNumberOfHits_track) tracker_purity = static_cast<double>(tracker_nshared);

	  if ((!AbsoluteNumberOfHits_track) && tracker_purity <= PurityCut_track) trackerOk = false;
	  
	  //if a track has just 3 hits in the Tracker we require that all 3 hits are shared
	  if (ThreeHitTracksAreSpecial && n_tracker_selected_hits==3 && tracker_nshared<3) trackerOk = false;
	}
	
	bool muonOk = false;
	if (n_muon_selected_hits != 0) {
	  if (muon_quality > muon_quality_cut) muonOk = true;
	  
	  muon_purity = static_cast<double>(muon_nshared)/static_cast<double>(n_muon_selected_hits);
	  if (AbsoluteNumberOfHits_muon) muon_purity = static_cast<double>(muon_nshared);

	  if ((!AbsoluteNumberOfHits_muon) &&  muon_purity <= PurityCut_muon) muonOk = false;
	}

	// (matchOk) has to account for different track types (tracker-only, standalone muons, global muons)
	bool matchOk = trackerOk || muonOk;

	// only for global muons: match both tracker and muon stub unless (acceptOneStubMatchings==true)
	if (!acceptOneStubMatchings && n_tracker_selected_hits!=0 && n_muon_selected_hits!=0)
	  matchOk = trackerOk && muonOk;
	
	if (matchOk) {
	  
          outputCollection[tpindex].push_back(IndexMatch(tindex,global_quality));
	  any_trackingParticle_matched = true;
	  
	  edm::LogVerbatim("MuonAssociatorByHitsHelper")
	    <<"************************************************************************************************************************"  
	    <<"\n"<< "TrackingParticle " << tpindex <<", q = "<<(*trpart).charge()<<", p = "<<(*trpart).p()
	    <<", pT = "<<(*trpart).pt()<<", eta = "<<(*trpart).eta()<<", phi = "<<(*trpart).phi()
	    <<"\n"<<" pdg code = "<<(*trpart).pdgId()
	    <<", made of "<<(*trpart).numberOfHits()<<" PSimHits, recounted "<<n_global_simhits<<" PSimHits"
	    <<" (tracker:"<<n_tracker_recounted_simhits<<"/muons:"<<n_muon_simhits<<")"
	    <<", from "<<(*trpart).g4Tracks().size()<<" SimTrack:";
	  for(TrackingParticle::g4t_iterator g4T=(*trpart).g4Track_begin(); 
	      g4T!=(*trpart).g4Track_end(); 
	      ++g4T) {
	    edm::LogVerbatim("MuonAssociatorByHitsHelper")
	      <<" Id:"<<(*g4T).trackId()<<"/Evt:("<<(*g4T).eventId().event()<<","<<(*g4T).eventId().bunchCrossing()<<")";
	  }
	  edm::LogVerbatim("MuonAssociatorByHitsHelper")
	    <<"\t selected "<<n_global_selected_simhits<<" PSimHits"
	    <<" (tracker:"<<n_tracker_selected_simhits<<"/muons:"<<n_muon_selected_simhits<<")"
	    << "\n\t **MATCHED** with quality = "<<global_quality<<" (tracker: "<<tracker_quality<<" / muon: "<<muon_quality<<")"
	    << "\n\t               and purity = "<<global_purity<<" (tracker: "<<tracker_purity<<" / muon: "<<muon_purity<<")"
	    << "\n\t   N shared hits = "<<global_nshared<<" (tracker: "<<tracker_nshared<<" / muon: "<<muon_nshared<<")"
	    <<"\n" <<"   to: reco::Track "<<tindex<<ZeroHitMuon
	    <<"\n\t"<< " made of "<<n_selected_hits<<" RecHits (tracker:"<<n_tracker_valid<<"/muons:"<<n_muon_selected_hits<<")";
	}
	else {
	  // print something only if this TrackingParticle shares some hits with the current reco::Track
	  if (global_nshared != 0) {
	    if (printRtS) edm::LogVerbatim("MuonAssociatorByHitsHelper")
	      <<"************************************************************************************************************************"  
	      <<"\n"<<"TrackingParticle " << tpindex <<", q = "<<(*trpart).charge()<<", p = "<<(*trpart).p()
	      <<", pT = "<<(*trpart).pt()<<", eta = "<<(*trpart).eta()<<", phi = "<<(*trpart).phi()
	      <<"\n"<<" pdg code = "<<(*trpart).pdgId()
	      <<", made of "<<(*trpart).numberOfHits()<<" PSimHits, recounted "<<n_global_simhits<<" PSimHits"
	      <<" (tracker:"<<n_tracker_recounted_simhits<<"/muons:"<<n_muon_simhits<<")"
	      <<", from "<<(*trpart).g4Tracks().size()<<" SimTrack:";
	    for(TrackingParticle::g4t_iterator g4T=(*trpart).g4Track_begin(); 
		g4T!=(*trpart).g4Track_end(); 
		++g4T) {
	      if (printRtS) edm::LogVerbatim("MuonAssociatorByHitsHelper")
		<<" Id:"<<(*g4T).trackId()<<"/Evt:("<<(*g4T).eventId().event()<<","<<(*g4T).eventId().bunchCrossing()<<")";
	    }
	    if (printRtS) edm::LogVerbatim("MuonAssociatorByHitsHelper")
	      <<"\t selected "<<n_global_selected_simhits<<" PSimHits"
	      <<" (tracker:"<<n_tracker_selected_simhits<<"/muons:"<<n_muon_selected_simhits<<")"
	      <<"\n\t NOT matched  to reco::Track "<<tindex<<ZeroHitMuon
	      <<" with quality = "<<global_quality<<" (tracker: "<<tracker_quality<<" / muon: "<<muon_quality<<")"
	      <<"\n\t and purity = "<<global_purity<<" (tracker: "<<tracker_purity<<" / muon: "<<muon_purity<<")"
	      <<"\n\t     N shared hits = "<<global_nshared<<" (tracker: "<<tracker_nshared<<" / muon: "<<muon_nshared<<")";
	  }
	}
      }  // loop over TrackingParticle's
    }   // if(n_matching_simhits != 0)
  }    // loop over reco Tracks
  
  if (!any_trackingParticle_matched) {
    edm::LogVerbatim("MuonAssociatorByHitsHelper")
      <<"\n"
      <<"************************************************************************************************************************"
      << "\n NO TrackingParticle associated to ANY input reco::Track ! \n"
      <<"************************************************************************************************************************"<<"\n";  
  } else {
    edm::LogVerbatim("MuonAssociatorByHitsHelper")
      <<"************************************************************************************************************************"<<"\n";  
  }
  
  for (IndexAssociation::iterator it = outputCollection.begin(), ed = outputCollection.end(); it != ed; ++it) {
    std::sort(it->second.begin(), it->second.end());
  }
  return outputCollection;
}


void MuonAssociatorByHitsHelper::getMatchedIds
(MapOfMatchedIds & tracker_matchedIds_valid, MapOfMatchedIds & muon_matchedIds_valid,
 MapOfMatchedIds & tracker_matchedIds_INVALID, MapOfMatchedIds & muon_matchedIds_INVALID,       
 int& n_tracker_valid, int& n_dt_valid, int& n_csc_valid, int& n_rpc_valid,
 int& n_tracker_matched_valid, int& n_dt_matched_valid, int& n_csc_matched_valid, int& n_rpc_matched_valid,
 int& n_tracker_INVALID, int& n_dt_INVALID, int& n_csc_INVALID, int& n_rpc_INVALID,
 int& n_tracker_matched_INVALID, int& n_dt_matched_INVALID, int& n_csc_matched_INVALID, int& n_rpc_matched_INVALID,
 trackingRecHit_iterator begin, trackingRecHit_iterator end,
 const TrackerHitAssociator* trackertruth, 
 const DTHitAssociator& dttruth, const CSCHitAssociator& csctruth, const RPCHitAssociator& rpctruth, bool printRtS,
 const TrackerTopology *tTopo) const

{
  tracker_matchedIds_valid.clear();
  muon_matchedIds_valid.clear();

  tracker_matchedIds_INVALID.clear();
  muon_matchedIds_INVALID.clear();

  n_tracker_valid = 0;
  n_dt_valid  = 0;
  n_csc_valid = 0;
  n_rpc_valid = 0;

  n_tracker_matched_valid = 0;
  n_dt_matched_valid  = 0;
  n_csc_matched_valid = 0;
  n_rpc_matched_valid = 0;
  
  n_tracker_INVALID = 0;
  n_dt_INVALID  = 0;
  n_csc_INVALID = 0;
  n_rpc_INVALID = 0;
  
  n_tracker_matched_INVALID = 0;
  n_dt_matched_INVALID  = 0;
  n_csc_matched_INVALID = 0;
  n_rpc_matched_INVALID = 0;
  
  std::vector<SimHitIdpr> SimTrackIds;

  // main loop on TrackingRecHits
  int iloop = 0;
  int iH = -1;
  for (trackingRecHit_iterator it = begin;  it != end; it++, iloop++) {
    stringstream hit_index;     
    hit_index<<iloop;

    const TrackingRecHit * hitp = getHitPtr(it);
    DetId geoid = hitp->geographicalId();    

    unsigned int detid = geoid.rawId();    
    stringstream detector_id;
    detector_id<<detid;

    string hitlog = "TrackingRecHit "+hit_index.str();
    string wireidlog;
    std::vector<string> DTSimHits;
    
    DetId::Detector det = geoid.det();
    int subdet = geoid.subdetId();
    
    bool valid_Hit = hitp->isValid();
    
    // Si-Tracker Hits
    if (det == DetId::Tracker && UseTracker) {
      stringstream detector_id;
      detector_id<< tTopo->print(detid);
      
      if (valid_Hit) hitlog = hitlog+" -Tracker - detID = "+detector_id.str();
      else hitlog = hitlog+" *** INVALID ***"+" -Tracker - detID = "+detector_id.str();
      
      iH++;
      SimTrackIds = trackertruth->associateHitId(*hitp);

      if (valid_Hit) {
	n_tracker_valid++;
	
	if(!SimTrackIds.empty()) {
	  n_tracker_matched_valid++;
	  //tracker_matchedIds_valid[iH] = SimTrackIds;
          tracker_matchedIds_valid.push_back( new uint_SimHitIdpr_pair(iH, SimTrackIds));
	}
      } else {
	n_tracker_INVALID++;

	if(!SimTrackIds.empty()) {
	  n_tracker_matched_INVALID++;
	  //tracker_matchedIds_INVALID[iH] = SimTrackIds;
          tracker_matchedIds_INVALID.push_back( new uint_SimHitIdpr_pair(iH, SimTrackIds));
	}
      }
    }  
    // Muon detector Hits    
    else if (det == DetId::Muon && UseMuon) {

      // DT Hits      
      if (subdet == MuonSubdetId::DT) {    
	DTWireId dtdetid = DTWireId(detid);
	stringstream dt_detector_id;
	dt_detector_id << dtdetid;
	if (valid_Hit) hitlog = hitlog+" -Muon DT - detID = "+dt_detector_id.str();	  
	else hitlog = hitlog+" *** INVALID ***"+" -Muon DT - detID = "+dt_detector_id.str();	  
	
	const DTRecHit1D * dtrechit = dynamic_cast<const DTRecHit1D *>(hitp);
	
	// single DT hits
	if (dtrechit) {
	  iH++;
	  SimTrackIds = dttruth.associateDTHitId(dtrechit);  
	  
	  if (valid_Hit) {
	    n_dt_valid++;

	    if (!SimTrackIds.empty()) {
	      n_dt_matched_valid++;
	      //muon_matchedIds_valid[iH] = SimTrackIds;
              muon_matchedIds_valid.push_back (new uint_SimHitIdpr_pair(iH,SimTrackIds));
	    }
	  } else {
	    n_dt_INVALID++;
	    
	    if (!SimTrackIds.empty()) {
	      n_dt_matched_INVALID++;
	      //muon_matchedIds_INVALID[iH] = SimTrackIds;
              muon_matchedIds_INVALID.push_back (new uint_SimHitIdpr_pair(iH,SimTrackIds));
	    }
	  }

	  if (dumpDT) {
	    DTWireId wireid = dtrechit->wireId();
	    stringstream wid; 
	    wid<<wireid;
	    std::vector<PSimHit> dtSimHits = dttruth.associateHit(*hitp);
	    
	    stringstream ndthits;
	    ndthits<<dtSimHits.size();
	    wireidlog = "\t DTWireId :"+wid.str()+", "+ndthits.str()+" associated PSimHit :";
	    
	    for (unsigned int j=0; j<dtSimHits.size(); j++) {
	      stringstream index;
	      index<<j;
	      stringstream simhit;
	      simhit<<dtSimHits[j];
	      string simhitlog = "\t\t PSimHit "+index.str()+": "+simhit.str();
	      DTSimHits.push_back(simhitlog);
	    }
	  }  // if (dumpDT)
	}

	// DT segments	
	else {
	  const DTRecSegment4D * dtsegment = dynamic_cast<const DTRecSegment4D *>(hitp);
	  
	  if (dtsegment) {
	    
	    std::vector<const TrackingRecHit *> componentHits, phiHits, zHits;
	    if (dtsegment->hasPhi()) {
	      phiHits = dtsegment->phiSegment()->recHits();
	      componentHits.insert(componentHits.end(),phiHits.begin(),phiHits.end());
	    }
	    if (dtsegment->hasZed()) {
	      zHits = dtsegment->zSegment()->recHits();
	      componentHits.insert(componentHits.end(),zHits.begin(),zHits.end());
	    }
	    if (printRtS) edm::LogVerbatim("MuonAssociatorByHitsHelper")
	      <<"\n\t this TrackingRecHit is a DTRecSegment4D with "
	      <<componentHits.size()<<" hits (phi:"<<phiHits.size()<<", z:"<<zHits.size()<<")";
	    
	    std::vector<SimHitIdpr> i_SimTrackIds;
	    int i_compHit = 0;
	    for (std::vector<const TrackingRecHit *>::const_iterator ithit =componentHits.begin(); 
		 ithit != componentHits.end(); ++ithit) {
	      i_compHit++;
	      
	      const DTRecHit1D * dtrechit1D = dynamic_cast<const DTRecHit1D *>(*ithit);
	      
	      i_SimTrackIds.clear();
	      if (dtrechit1D) {
		iH++;
		i_SimTrackIds = dttruth.associateDTHitId(dtrechit1D);  
		
		if (valid_Hit) {
		  // validity check is on the segment, but hits are counted one-by-one
		  n_dt_valid++; 

		  if (!i_SimTrackIds.empty()) {
		    n_dt_matched_valid++;
		    //muon_matchedIds_valid[iH] = i_SimTrackIds;
                    muon_matchedIds_valid.push_back (new uint_SimHitIdpr_pair(iH,i_SimTrackIds));
		  }
		} else {
		  n_dt_INVALID++;
		  
		  if (!i_SimTrackIds.empty()) {
		    n_dt_matched_INVALID++;
		    //muon_matchedIds_INVALID[iH] = i_SimTrackIds;
                    muon_matchedIds_INVALID.push_back (new uint_SimHitIdpr_pair(iH,i_SimTrackIds));
                    
		  } 
		}
	      } else if (printRtS) edm::LogWarning("MuonAssociatorByHitsHelper")
		<<"*** WARNING in MuonAssociatorByHitsHelper::getMatchedIds, null dynamic_cast of a DT TrackingRecHit !";
	      
	      unsigned int i_detid = (*ithit)->geographicalId().rawId();
	      DTWireId i_dtdetid = DTWireId(i_detid);

	      stringstream i_dt_detector_id;
	      i_dt_detector_id << i_dtdetid;

	      stringstream i_ss;
	      i_ss<<"\t\t hit "<<i_compHit<<" -Muon DT - detID = "<<i_dt_detector_id.str();

	      string i_hitlog = i_ss.str();
	      i_hitlog = i_hitlog + write_matched_simtracks(i_SimTrackIds);
	      if (printRtS) edm::LogVerbatim("MuonAssociatorByHitsHelper") << i_hitlog;
	      
	      SimTrackIds.insert(SimTrackIds.end(),i_SimTrackIds.begin(),i_SimTrackIds.end());
	    }	      
	  }  // if (dtsegment)

	  else if (printRtS) edm::LogWarning("MuonAssociatorByHitsHelper")
	    <<"*** WARNING in MuonAssociatorByHitsHelper::getMatchedIds, DT TrackingRecHit is neither DTRecHit1D nor DTRecSegment4D ! ";	    
	}
      }
      
      // CSC Hits
      else if (subdet == MuonSubdetId::CSC) {
	CSCDetId cscdetid = CSCDetId(detid);
	stringstream csc_detector_id;
	csc_detector_id << cscdetid;
	if (valid_Hit) hitlog = hitlog+" -Muon CSC- detID = "+csc_detector_id.str();	  
	else hitlog = hitlog+" *** INVALID ***"+" -Muon CSC- detID = "+csc_detector_id.str();	  
	
	const CSCRecHit2D * cscrechit = dynamic_cast<const CSCRecHit2D *>(hitp);
	
	// single CSC hits
	if (cscrechit) {
	  iH++;
	  SimTrackIds = csctruth.associateCSCHitId(cscrechit);
	  
	  if (valid_Hit) {
	    n_csc_valid++;
	    
	    if (!SimTrackIds.empty()) {
	      n_csc_matched_valid++;
	      //muon_matchedIds_valid[iH] = SimTrackIds;
              muon_matchedIds_valid.push_back (new uint_SimHitIdpr_pair(iH,SimTrackIds));
	    }
	  } else {
	    n_csc_INVALID++;
	    
	    if (!SimTrackIds.empty()) {
	      n_csc_matched_INVALID++;
	      //muon_matchedIds_INVALID[iH] = SimTrackIds;
              muon_matchedIds_INVALID.push_back (new uint_SimHitIdpr_pair(iH,SimTrackIds));
	    }
	  }
	}
	
	// CSC segments
	else {
	  const CSCSegment * cscsegment = dynamic_cast<const CSCSegment *>(hitp);
	  
	  if (cscsegment) {
	    
	    std::vector<const TrackingRecHit *> componentHits = cscsegment->recHits();
	    if (printRtS) edm::LogVerbatim("MuonAssociatorByHitsHelper")
	      <<"\n\t this TrackingRecHit is a CSCSegment with "<<componentHits.size()<<" hits";
	    
	    std::vector<SimHitIdpr> i_SimTrackIds;
	    int i_compHit = 0;
	    for (std::vector<const TrackingRecHit *>::const_iterator ithit =componentHits.begin(); 
		 ithit != componentHits.end(); ++ithit) {
	      i_compHit++;
	      
	      const CSCRecHit2D * cscrechit2D = dynamic_cast<const CSCRecHit2D *>(*ithit);
	      
	      i_SimTrackIds.clear();
	      if (cscrechit2D) {
		iH++;
		i_SimTrackIds = csctruth.associateCSCHitId(cscrechit2D);

		if (valid_Hit) {
		  // validity check is on the segment, but hits are counted one-by-one
		  n_csc_valid++;

		  if (!i_SimTrackIds.empty()) {
		    n_csc_matched_valid++;
		    //muon_matchedIds_valid[iH] =  i_SimTrackIds;
                    muon_matchedIds_valid.push_back (new uint_SimHitIdpr_pair(iH,i_SimTrackIds));
		  }
		} else {
		  n_csc_INVALID++;
		  
		  if (!i_SimTrackIds.empty()) {
		    n_csc_matched_INVALID++;
		    //muon_matchedIds_INVALID[iH] =  i_SimTrackIds;
                    muon_matchedIds_INVALID.push_back (new uint_SimHitIdpr_pair(iH,i_SimTrackIds));
		  }
		}
	      } else if (printRtS) edm::LogWarning("MuonAssociatorByHitsHelper")
		<<"*** WARNING in MuonAssociatorByHitsHelper::getMatchedIds, null dynamic_cast of a CSC TrackingRecHit !";
	      
	      unsigned int i_detid = (*ithit)->geographicalId().rawId();
	      CSCDetId i_cscdetid = CSCDetId(i_detid);

	      stringstream i_csc_detector_id;
	      i_csc_detector_id << i_cscdetid;

	      stringstream i_ss;
	      i_ss<<"\t\t hit "<<i_compHit<<" -Muon CSC- detID = "<<i_csc_detector_id.str();

	      string i_hitlog = i_ss.str();
	      i_hitlog = i_hitlog + write_matched_simtracks(i_SimTrackIds);
	      if (printRtS) edm::LogVerbatim("MuonAssociatorByHitsHelper") << i_hitlog;
	      
	      SimTrackIds.insert(SimTrackIds.end(),i_SimTrackIds.begin(),i_SimTrackIds.end());
	    }	    
	  }  // if (cscsegment)

	  else if (printRtS) edm::LogWarning("MuonAssociatorByHitsHelper")
	    <<"*** WARNING in MuonAssociatorByHitsHelper::getMatchedIds, CSC TrackingRecHit is neither CSCRecHit2D nor CSCSegment ! ";
	}
      }
      
      // RPC Hits
      else if (subdet == MuonSubdetId::RPC) {
	RPCDetId rpcdetid = RPCDetId(detid);
	stringstream rpc_detector_id;
	rpc_detector_id << rpcdetid;
	if (valid_Hit) hitlog = hitlog+" -Muon RPC- detID = "+rpc_detector_id.str();	  
	else hitlog = hitlog+" *** INVALID ***"+" -Muon RPC- detID = "+rpc_detector_id.str();	  
	
	iH++;
	SimTrackIds = rpctruth.associateRecHit(*hitp);
	
	if (valid_Hit) {
	  n_rpc_valid++;

	  if (!SimTrackIds.empty()) {
	    n_rpc_matched_valid++;
	    //muon_matchedIds_valid[iH] = SimTrackIds;
            muon_matchedIds_valid.push_back (new uint_SimHitIdpr_pair(iH,SimTrackIds));
            
	  }
	} else {
	  n_rpc_INVALID++;
	  
	  if (!SimTrackIds.empty()) {
	    n_rpc_matched_INVALID++;
	    //muon_matchedIds_INVALID[iH] = SimTrackIds;
            muon_matchedIds_INVALID.push_back (new uint_SimHitIdpr_pair(iH,SimTrackIds));
	  }
	}
	
      } else if (printRtS) edm::LogVerbatim("MuonAssociatorByHitsHelper")
	<<"TrackingRecHit "<<iloop<<"  *** WARNING *** Unexpected Hit from Detector = "<<det;
    }
    else continue;
    
    hitlog = hitlog + write_matched_simtracks(SimTrackIds);
    
    if (printRtS) edm::LogVerbatim("MuonAssociatorByHitsHelper") << hitlog;
    if (printRtS && dumpDT && det==DetId::Muon && subdet==MuonSubdetId::DT) {
      edm::LogVerbatim("MuonAssociatorByHitsHelper") <<wireidlog;
      for (unsigned int j=0; j<DTSimHits.size(); j++) {
	edm::LogVerbatim("MuonAssociatorByHitsHelper") <<DTSimHits[j];
      }
    }
    
  } //trackingRecHit loop
}

int MuonAssociatorByHitsHelper::getShared(MapOfMatchedIds & matchedIds, TrackingParticleCollection::const_iterator trpart) const {
  int nshared = 0;


  // map is indexed over the rechits of the reco::Track (no double-countings allowed)
  for (MapOfMatchedIds::const_iterator iRecH=matchedIds.begin(); iRecH!=matchedIds.end(); ++iRecH) {

    // vector of associated simhits associated to the current rechit
    std::vector<SimHitIdpr> const & SimTrackIds = (*iRecH).second;
    
    bool found = false;
    
    for (std::vector<SimHitIdpr>::const_iterator iSimH=SimTrackIds.begin(); iSimH!=SimTrackIds.end(); ++iSimH) {
      uint32_t simtrackId = iSimH->first;
      EncodedEventId evtId = iSimH->second;
      
      // look for shared hits with the given TrackingParticle (looping over component SimTracks)
      for (TrackingParticle::g4t_iterator simtrack = trpart->g4Track_begin(); simtrack !=  trpart->g4Track_end(); ++simtrack) {
	if (simtrack->trackId() == simtrackId  &&  simtrack->eventId() == evtId) {
	  found = true;
	  break;
	}
      }
      
      if (found) {
	nshared++;
	break;
      }
    }
  }
  
  return nshared;
}

std::string MuonAssociatorByHitsHelper::write_matched_simtracks(const std::vector<SimHitIdpr>& SimTrackIds) const {
  if (SimTrackIds.empty())
    return "  *** UNMATCHED ***";

  string hitlog(" matched to SimTrack");
  
  for(size_t j=0; j<SimTrackIds.size(); j++)
  {
    char buf[64];
    snprintf(buf, 64, " Id:%i/Evt:(%i,%i) ", SimTrackIds[j].first, SimTrackIds[j].second.event(), SimTrackIds[j].second.bunchCrossing());
    hitlog += buf;
  }
  return hitlog;
}

