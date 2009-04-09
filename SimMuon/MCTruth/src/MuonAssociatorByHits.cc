#include "SimMuon/MCTruth/interface/MuonAssociatorByHits.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h" 
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include <sstream>

using namespace reco;
using namespace std;

MuonAssociatorByHits::MuonAssociatorByHits (const edm::ParameterSet& conf) :  
  AbsoluteNumberOfHits_track(conf.getParameter<bool>("AbsoluteNumberOfHits_track")),
  MinHitCut_track(conf.getParameter<unsigned int>("MinHitCut_track")),
  AbsoluteNumberOfHits_muon(conf.getParameter<bool>("AbsoluteNumberOfHits_muon")),
  MinHitCut_muon(conf.getParameter<unsigned int>("MinHitCut_muon")),
  UseTracker(conf.getParameter<bool>("UseTracker")),
  UseMuon(conf.getParameter<bool>("UseMuon")),
  PurityCut_track(conf.getParameter<double>("PurityCut_track")),
  PurityCut_muon(conf.getParameter<double>("PurityCut_muon")),
  EfficiencyCut_track(conf.getParameter<double>("EfficiencyCut_track")),
  EfficiencyCut_muon(conf.getParameter<double>("EfficiencyCut_muon")),
  UsePixels(conf.getParameter<bool>("UsePixels")),
  UseGrouped(conf.getParameter<bool>("UseGrouped")),
  UseSplitting(conf.getParameter<bool>("UseSplitting")),
  ThreeHitTracksAreSpecial(conf.getParameter<bool>("ThreeHitTracksAreSpecial")),
  dumpDT(conf.getParameter<bool>("dumpDT")),
  dumpInputCollections(conf.getParameter<bool>("dumpInputCollections")),
  crossingframe(conf.getParameter<bool>("crossingframe")),
  simtracksTag(conf.getParameter<edm::InputTag>("simtracksTag")),
  simtracksXFTag(conf.getParameter<edm::InputTag>("simtracksXFTag")),
  conf_(conf)
{
  edm::LogVerbatim("MuonAssociatorByHits") << "constructing  MuonAssociatorByHits" << conf_.dump();
  edm::InputTag tracksTag = conf.getParameter<edm::InputTag>("tracksTag");
  edm::InputTag tpTag = conf.getParameter<edm::InputTag>("tpTag");
  edm::LogVerbatim("MuonAssociatorByHits") << "\n MuonAssociatorByHits will associate reco::Tracks with "<<tracksTag
					   << "\n\t\t and TrackingParticles with "<<tpTag;
  const std::string recoTracksLabel = tracksTag.label();

  // check and fix inconsistent input settings
  // tracks with hits only on muon detectors
  if (recoTracksLabel == "standAloneMuons" || recoTracksLabel == "hltL2Muons") {
    if (UseTracker) {
      edm::LogWarning("MuonAssociatorByHits") 
	<<"\n*** WARNING : inconsistent input tracksTag = "<<tracksTag
	<<"\n with UseTracker = true"<<"\n ---> setting UseTracker = false ";
      UseTracker = false;
    }
    if (!UseMuon) {
      edm::LogWarning("MuonAssociatorByHits") 
	<<"\n*** WARNING : inconsistent input tracksTag = "<<tracksTag
	<<"\n with UseMuon = false"<<"\n ---> setting UseMuon = true ";
      UseMuon = true;
    }
  }
  // tracks with hits only on tracker
  if (recoTracksLabel == "generalTracks") {
    if (UseMuon) {
      edm::LogWarning("MuonAssociatorByHits") 
	<<"\n*** WARNING : inconsistent input tracksTag = "<<tracksTag
	<<"\n with UseMuon = true"<<"\n ---> setting UseMuon = false ";
      UseMuon = false;
    }
    if (!UseTracker) {
      edm::LogWarning("MuonAssociatorByHits") 
	<<"\n*** WARNING : inconsistent input tracksTag = "<<tracksTag
	<<"\n with UseTracker = false"<<"\n ---> setting UseTracker = true ";
      UseTracker = true;
    }
  }
  
  // up to the user in the other cases - print a message
  if (UseTracker) edm::LogVerbatim("MuonAssociatorByHits")<<"\n UseTracker = TRUE  : Tracker SimHits and RecHits WILL be counted";
  else edm::LogVerbatim("MuonAssociatorByHits") <<"\n UseTracker = FALSE : Tracker SimHits and RecHits WILL NOT be counted";
  
  // up to the user in the other cases - print a message
  if (UseMuon) edm::LogVerbatim("MuonAssociatorByHits")<<" UseMuon = TRUE  : Muon SimHits and RecHits WILL be counted";
  else edm::LogVerbatim("MuonAssociatorByHits") <<" UseMuon = FALSE : Muon SimHits and RecHits WILL NOT be counted"<<endl;
  
}

MuonAssociatorByHits::~MuonAssociatorByHits()
{
}

RecoToSimCollection  
MuonAssociatorByHits::associateRecoToSim(edm::RefToBaseVector<reco::Track>& tC,
					  edm::RefVector<TrackingParticleCollection>& TPCollectionH,
					  const edm::Event * e, const edm::EventSetup * setup) const{

  int tracker_nshared = 0;
  int muon_nshared = 0;
  int global_nshared = 0;

  double tracker_quality = 0;
  double tracker_quality_cut;
  if (AbsoluteNumberOfHits_track) tracker_quality_cut = static_cast<double>(MinHitCut_track); 
  else tracker_quality_cut = PurityCut_track;

  double muon_quality = 0;
  double muon_quality_cut;
  if (AbsoluteNumberOfHits_muon) muon_quality_cut = static_cast<double>(MinHitCut_muon); 
  else muon_quality_cut = PurityCut_muon;

  double global_quality = 0;
  
  std::vector< SimHitIdpr> tracker_matchedIds, muon_matchedIds;

  RecoToSimCollection  outputCollection;
  bool printRtS(true);

  // Tracker hit association  
  TrackerHitAssociator * trackertruth = new TrackerHitAssociator::TrackerHitAssociator(*e, conf_);
  // CSC hit association
  MuonTruth csctruth(*e,*setup,conf_);
  // DT hit association
  printRtS = true;
  DTHitAssociator dttruth(*e,*setup,conf_,printRtS);
  // RPC hit association
  RPCHitAssociator rpctruth(*e,*setup,conf_);
  
  TrackingParticleCollection tPC;
  if (TPCollectionH.size()!=0) tPC = *(TPCollectionH.product());

  if (dumpInputCollections) {
    // reco::Track collection
    edm::LogVerbatim("MuonAssociatorByHits")<<"\n"<<"reco::Track collection --- size = "<<tC.size();
    int i = 0;
    for (edm::RefToBaseVector<reco::Track>::const_iterator ITER=tC.begin(); ITER!=tC.end(); ITER++, i++) {
      edm::LogVerbatim("MuonAssociatorByHits")
	<<"Track "<<i<<" : p = "<<(*ITER)->p()<<", charge = "<<(*ITER)->charge()
	<<", pT = "<<(*ITER)->pt()<<", eta = "<<(*ITER)->eta()<<", phi = "<<(*ITER)->phi();    
    }

    // TrackingParticle collection
    edm::LogVerbatim("MuonAssociatorByHits")<<"\n"<<"TrackingParticle collection --- size = "<<tPC.size();
    int j = 0;
    for(TrackingParticleCollection::const_iterator ITER=tPC.begin(); ITER!=tPC.end(); ITER++, j++) {
      edm::LogVerbatim("MuonAssociatorByHits")
	<<"TrackingParticle "<<j<<", q = "<<ITER->charge()<<", p = "<<ITER->p()
	<<", pT = "<<ITER->pt()<<", eta = "<<ITER->eta()<<", phi = "<<ITER->phi();
      
      edm::LogVerbatim("MuonAssociatorByHits")
	<<"\t pdg code = "<<ITER->pdgId()<<", made of "<<ITER->trackPSimHit().size()<<" PSimHit"
	<<" (in "<<ITER->matchedHit()<<" layers)"
	<<" from "<<ITER->g4Tracks().size()<<" SimTrack:";
      for (TrackingParticle::g4t_iterator g4T=ITER->g4Track_begin(); g4T!=ITER->g4Track_end(); g4T++) {
	edm::LogVerbatim("MuonAssociatorByHits")
	  <<"\t\t Id:"<<g4T->trackId()<<"/Evt:("<<g4T->eventId().event()<<","<<g4T->eventId().bunchCrossing()<<")";
      }    
    }

    // SimTrack collection
    edm::Handle<CrossingFrame<SimTrack> > cf_simtracks;
    edm::Handle<edm::SimTrackContainer> simTrackCollection;
    
    if (crossingframe) {
      e->getByLabel(simtracksXFTag,cf_simtracks);
      auto_ptr<MixCollection<SimTrack> > SimTk( new MixCollection<SimTrack>(cf_simtracks.product()) );
      edm::LogVerbatim("MuonAssociatorByHits")<<"\n"<<"CrossingFrame<SimTrack> collection with InputTag = "<<simtracksXFTag
					      <<" has size = "<<SimTk->size();
      int k = 0;
      for (MixCollection<SimTrack>::MixItr ITER=SimTk->begin(); ITER!=SimTk->end(); ITER++, k++) {
	edm::LogVerbatim("MuonAssociatorByHits")
	  <<"SimTrack "<<k
	  <<" - Id:"<<ITER->trackId()<<"/Evt:("<<ITER->eventId().event()<<","<<ITER->eventId().bunchCrossing()<<")"
	  <<" pdgId = "<<ITER->type()<<", q = "<<ITER->charge()<<", p = "<<ITER->momentum().P()
	  <<", pT = "<<ITER->momentum().Pt()<<", eta = "<<ITER->momentum().Eta()<<", phi = "<<ITER->momentum().Phi();
      }
    }
    else {
      e->getByLabel(simtracksTag,simTrackCollection);
      const edm::SimTrackContainer simTC = *(simTrackCollection.product());
      edm::LogVerbatim("MuonAssociatorByHits")<<"\n"<<"SimTrack collection with InputTag = "<<simtracksTag
					      <<" has size = "<<simTC.size()<<endl;
      int k = 0;
      for(edm::SimTrackContainer::const_iterator ITER=simTC.begin(); ITER!=simTC.end(); ITER++, k++){
	edm::LogVerbatim("MuonAssociatorByHits")
	  <<"SimTrack "<<k
	  <<" - Id:"<<ITER->trackId()<<"/Evt:("<<ITER->eventId().event()<<","<<ITER->eventId().bunchCrossing()<<")"
	  <<" pdgId = "<<ITER->type()<<", q = "<<ITER->charge()<<", p = "<<ITER->momentum().P()
	  <<", pT = "<<ITER->momentum().Pt()<<", eta = "<<ITER->momentum().Eta()<<", phi = "<<ITER->momentum().Phi();
      }
    }
  }
  
  int tindex=0;
  for (edm::RefToBaseVector<reco::Track>::const_iterator track=tC.begin(); track!=tC.end(); track++, tindex++) {
    edm::LogVerbatim("MuonAssociatorByHits")
      <<"\n"<<"reco::Track "<<tindex
      <<", pT = " << (*track)->pt()<<", eta = "<<(*track)->momentum().eta()<<", phi = "<<(*track)->momentum().phi()
      <<", number of RecHits = "<<(*track)->recHitsSize()<<" ("<<(*track)->found()<<" valid hits, "<<(*track)->lost()<<" lost hits) \n";
    
    tracker_matchedIds.clear();
    muon_matchedIds.clear();

    bool this_track_matched = false;
    int n_matching_simhits = 0;

    int n_valid_hits = 0;          // number of valid hits      (Total)
    int n_tracker_valid_hits = 0;  //                           (Tracker)
    int n_muon_valid_hits = 0;     //                           (DT+CSC+RPC)
    int n_dt_valid_hits = 0;       //                           (DT)
    int n_csc_valid_hits = 0;      //                           (CSC)
    int n_rpc_valid_hits = 0;      //                           (RPC)

    int n_selected_hits = 0;          // number of selected hits   (Total)
    int n_tracker_selected_hits = 0;  //                           (Tracker)
    int n_muon_selected_hits = 0;     //                           (DT+CSC+RPC)
    int n_dt_selected_hits = 0;       //                           (DT)
    int n_csc_selected_hits = 0;      //                           (CSC)
    int n_rpc_selected_hits = 0;      //                           (RPC)

    int n_matched_hits = 0;          // number of matched hits    (Total)
    int n_tracker_matched_hits = 0;  //                           (Tracker)
    int n_muon_matched_hits = 0;     //                           (DT+CSC+RPC)
    int n_dt_matched_hits = 0;       //                           (DT)
    int n_csc_matched_hits = 0;      //                           (CSC)
    int n_rpc_matched_hits = 0;      //                           (RPC)

    printRtS = true;
    getMatchedIds<trackingRecHit_iterator>(tracker_matchedIds, muon_matchedIds,
					   n_valid_hits, n_tracker_valid_hits, n_dt_valid_hits, n_csc_valid_hits, n_rpc_valid_hits,
					   n_selected_hits, n_tracker_selected_hits, n_dt_selected_hits, n_csc_selected_hits, n_rpc_selected_hits,
					   n_matched_hits, n_tracker_matched_hits, n_dt_matched_hits, n_csc_matched_hits, n_rpc_matched_hits,
					   (*track)->recHitsBegin(), (*track)->recHitsEnd(), 
					   trackertruth, dttruth, csctruth, rpctruth, printRtS);

    n_matching_simhits = tracker_matchedIds.size() + muon_matchedIds.size(); 
    n_muon_valid_hits = n_dt_valid_hits + n_csc_valid_hits + n_rpc_valid_hits;
    n_muon_selected_hits = n_dt_selected_hits + n_csc_selected_hits + n_rpc_selected_hits;
    n_muon_matched_hits = n_dt_matched_hits + n_csc_matched_hits + n_rpc_matched_hits;

    edm::LogVerbatim("MuonAssociatorByHits")
      <<"\n"<<"##### all RecHits = "<<(*track)->recHitsSize()
      <<", # matching SimHits = " << n_matching_simhits <<" (may be more than one per rechit)"
      <<"\n"<< "# valid RecHits    = " <<n_valid_hits  <<" (" <<n_tracker_valid_hits<<"/"
      <<n_dt_valid_hits<<"/"<<n_csc_valid_hits<<"/"<<n_rpc_valid_hits<<" in Tracker/DT/CSC/RPC)"
      <<"\n"<< "# selected RecHits = " <<n_selected_hits  <<" (" <<n_tracker_selected_hits<<"/"
      <<n_dt_selected_hits<<"/"<<n_csc_selected_hits<<"/"<<n_rpc_selected_hits<<" in Tracker/DT/CSC/RPC)"
      <<"\n"<< "# matched RecHits  = " <<n_matched_hits<<" ("<<n_tracker_matched_hits<<"/"
      <<n_dt_matched_hits<<"/"<<n_csc_matched_hits<<"/"<<n_rpc_matched_hits<<" in Tracker/DT/CSC/RPC)";

    if (!n_matching_simhits) {
      edm::LogWarning("MuonAssociatorByHits")<<"*** WARNING: no matching PSimHit found for this reco::Track !";
    }

    if (n_valid_hits != (*track)->found()) {
      edm::LogWarning("MuonAssociatorByHits")
	<<"*** WARNING: Number of valid RecHits in this track should be:  track->found() = "<<(*track)->found()
        <<", but getMatchedIds finds:  n_valid_hits = "<<n_valid_hits;
    }

    std::vector<SimHitIdpr> tracker_idcachev, muon_idcachev;

    if(n_matching_simhits) {
      edm::LogVerbatim("MuonAssociatorByHits")
	<<"\n"<< "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
	<<"\n"<< "reco::Track "<<tindex<<", q = "<<(*track)->charge()<<", p = " << (*track)->p()<<", pT = " << (*track)->pt()
	<<", eta = "<<(*track)->momentum().eta()<<", phi = "<<(*track)->momentum().phi()
	<<"\n\t"<< "made of "<<n_valid_hits<<" valid RecHits (tracker:"<<n_tracker_valid_hits<<"/muons:"<<n_muon_valid_hits<<")"
	<<"\n\t\t"<<n_selected_hits<<" selected RecHits (tracker:"<<n_tracker_selected_hits<<"/muons:"<<n_muon_selected_hits<<")";

      int tpindex = 0;
      for (TrackingParticleCollection::const_iterator trpart = tPC.begin(); trpart != tPC.end(); ++trpart, ++tpindex) {
	tracker_idcachev.clear();
	tracker_nshared = getShared(tracker_matchedIds, tracker_idcachev, trpart);
	muon_idcachev.clear();
	muon_nshared = getShared(muon_matchedIds, muon_idcachev, trpart);
        global_nshared = tracker_nshared + muon_nshared;

	if (AbsoluteNumberOfHits_track) tracker_quality = static_cast<double>(tracker_nshared);
	else if(n_tracker_selected_hits!=0) tracker_quality = (static_cast<double>(tracker_nshared)/static_cast<double>(n_tracker_selected_hits));
	else tracker_quality = 0;

	if (AbsoluteNumberOfHits_muon) muon_quality = static_cast<double>(muon_nshared);
	else if(n_muon_selected_hits!=0) muon_quality = (static_cast<double>(muon_nshared)/static_cast<double>(n_muon_selected_hits));
	else muon_quality = 0;

	if(n_selected_hits!=0) global_quality = (static_cast<double>(global_nshared)/static_cast<double>(n_selected_hits));
	else global_quality = 0;

	bool matchOk = true;
	if (n_selected_hits==0) matchOk = false;
	else {
	  if (n_tracker_selected_hits != 0) {
	    if (tracker_quality < tracker_quality_cut) matchOk = false;
	    //if a track has just 3 hits in the Tracker we require that all 3 hits are shared
	    if (ThreeHitTracksAreSpecial && n_tracker_selected_hits==3 && tracker_nshared<3) matchOk = false;
	  }
	  
	  if (n_muon_selected_hits != 0) {
	    if (muon_quality < muon_quality_cut) matchOk = false;
	  }
	}
        
	if (matchOk) {

	  outputCollection.insert(tC[tindex],
				  std::make_pair(edm::Ref<TrackingParticleCollection>(TPCollectionH, tpindex),
						 global_quality));
	  this_track_matched = true;

	  edm::LogVerbatim("MuonAssociatorByHits")
	    << "\t\t"<< " **MATCHED** with quality = "<<global_quality<<" (tracker: "<<tracker_quality<<" / muon: "<<muon_quality<<") to:"
	    <<"\n"<< "TrackingParticle " <<tpindex<<", q = "<<(*trpart).charge()<<", p = "<<(*trpart).p()
	    <<", pT = "<<(*trpart).pt()<<", eta = "<<(*trpart).eta()<<", phi = "<<(*trpart).phi()
	    <<"\n\t"<< " pdg code = "<<(*trpart).pdgId()<<", made of "<<(*trpart).trackPSimHit().size()<<" PSimHits"
	    //	    <<" (in "<<(*trpart).matchedHit()<<" layers)"
	    <<" from "<<(*trpart).g4Tracks().size()<<" SimTrack:";
	  for(TrackingParticle::g4t_iterator g4T=(*trpart).g4Track_begin(); 
	      g4T!=(*trpart).g4Track_end(); 
	      ++g4T) {
	    edm::LogVerbatim("MuonAssociatorByHits")
	      <<"\t\t"<< " Id:"<<(*g4T).trackId()<<"/Evt:("<<(*g4T).eventId().event()<<","<<(*g4T).eventId().bunchCrossing()<<")";
	  }
	}
	else {
	  // print something only if this TrackingParticle shares some hits with the current reco::Track
	  if (global_nshared >0) 
	    edm::LogVerbatim("MuonAssociatorByHits")
	      <<"\t\t"<< " NOT matched to TrackingParticle "<<tpindex
	      << " with quality = "<<global_quality<<" (tracker: "<<tracker_quality<<" / muon: "<<muon_quality<<")";
	}
	
      }    //  loop over TrackingParticle

      if (!this_track_matched) {
	edm::LogVerbatim("MuonAssociatorByHits")
	  <<"\n"<<" NOT matched to any TrackingParticle";
      }
      
      edm::LogVerbatim("MuonAssociatorByHits")
	<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<"\n";
      
    }    //  if(!n_matching_simhits)
    
  }    // loop over reco::Track

  if (!tC.size()) 
    edm::LogVerbatim("MuonAssociatorByHits")<<"0 reconstructed tracks (-->> 0 associated !)";

  delete trackertruth;
  outputCollection.post_insert();
  return outputCollection;
}


SimToRecoCollection  
MuonAssociatorByHits::associateSimToReco(edm::RefToBaseVector<reco::Track>& tC, 
					  edm::RefVector<TrackingParticleCollection>& TPCollectionH,
					  const edm::Event * e, const edm::EventSetup * setup) const{

  int tracker_nshared = 0;
  int muon_nshared = 0;
  int global_nshared = 0;

  double tracker_quality = 0;
  double tracker_quality_cut;
  if (AbsoluteNumberOfHits_track) tracker_quality_cut = static_cast<double>(MinHitCut_track); 
  else tracker_quality_cut = EfficiencyCut_track;
  
  double muon_quality = 0;
  double muon_quality_cut;
  if (AbsoluteNumberOfHits_muon) muon_quality_cut = static_cast<double>(MinHitCut_muon); 
  else muon_quality_cut = EfficiencyCut_muon;

  double global_quality = 0;

  double tracker_purity = 0;
  double muon_purity = 0;
  double global_purity = 0;
  
  std::vector< SimHitIdpr> tracker_matchedIds, muon_matchedIds;

  SimToRecoCollection  outputCollection;

  bool printRtS(true);

  // Tracker hit association  
  TrackerHitAssociator * trackertruth = new TrackerHitAssociator::TrackerHitAssociator(*e, conf_);
  // CSC hit association
  MuonTruth csctruth(*e,*setup,conf_);
  // DT hit association
  printRtS = false;
  DTHitAssociator dttruth(*e,*setup,conf_,printRtS);  
  // RPC hit association
  RPCHitAssociator rpctruth(*e,*setup,conf_);
 
  TrackingParticleCollection tPC;
  if (TPCollectionH.size()!=0) tPC = *(TPCollectionH.product());

  bool any_trackingParticle_matched = false;
  
  int tindex=0;
  for (edm::RefToBaseVector<reco::Track>::const_iterator track=tC.begin(); track!=tC.end(); track++, tindex++) {
    if (printRtS) edm::LogVerbatim("MuonAssociatorByHits")
      <<"\n"<<"reco::Track "<<tindex
      <<", pT = " << (*track)->pt()<<", eta = "<<(*track)->momentum().eta()<<", phi = "<<(*track)->momentum().phi()
      <<", number of RecHits = "<<(*track)->recHitsSize()<<" ("<<(*track)->found()<<" valid hits, "<<(*track)->lost()<<" lost hits)";
    
    tracker_matchedIds.clear();
    muon_matchedIds.clear();
    int n_matching_simhits = 0;

    int n_valid_hits = 0;          // number of valid hits      (Total)
    int n_tracker_valid_hits = 0;  //                           (Tracker)
    int n_muon_valid_hits = 0;     //                           (DT+CSC+RPC)
    int n_dt_valid_hits = 0;       //                           (DT)
    int n_csc_valid_hits = 0;      //                           (CSC)
    int n_rpc_valid_hits = 0;      //                           (RPC)
    
    int n_selected_hits = 0;          // number of selected hits   (Total)
    int n_tracker_selected_hits = 0;  //                           (Tracker)
    int n_muon_selected_hits = 0;     //                           (DT+CSC+RPC)
    int n_dt_selected_hits = 0;       //                           (DT)
    int n_csc_selected_hits = 0;      //                           (CSC)
    int n_rpc_selected_hits = 0;      //                           (RPC)
    
    int n_matched_hits = 0;          // number of matched hits    (Total)
    int n_tracker_matched_hits = 0;  //                           (Tracker)
    int n_muon_matched_hits = 0;     //                           (DT+CSC+RPC)
    int n_dt_matched_hits = 0;       //                           (DT)
    int n_csc_matched_hits = 0;      //                           (CSC)
    int n_rpc_matched_hits = 0;      //                           (RPC)

    printRtS = false;
    getMatchedIds<trackingRecHit_iterator>(tracker_matchedIds, muon_matchedIds,
					   n_valid_hits, n_tracker_valid_hits, n_dt_valid_hits, n_csc_valid_hits, n_rpc_valid_hits,
					   n_selected_hits, n_tracker_selected_hits, n_dt_selected_hits, n_csc_selected_hits, n_rpc_selected_hits,
					   n_matched_hits, n_tracker_matched_hits, n_dt_matched_hits, n_csc_matched_hits, n_rpc_matched_hits,
					   (*track)->recHitsBegin(), (*track)->recHitsEnd(), 
					   trackertruth, dttruth, csctruth, rpctruth, printRtS);
    
    n_matching_simhits = tracker_matchedIds.size() + muon_matchedIds.size(); 
    n_muon_valid_hits = n_dt_valid_hits + n_csc_valid_hits + n_rpc_valid_hits;
    n_muon_selected_hits = n_dt_selected_hits + n_csc_selected_hits + n_rpc_selected_hits;
    n_muon_matched_hits = n_dt_matched_hits + n_csc_matched_hits + n_rpc_matched_hits;

    if (printRtS) edm::LogVerbatim("MuonAssociatorByHits")
      <<"\n"<<"*** # all RecHits = "<<(*track)->recHitsSize()
      <<", # matching PSimHit = " << n_matching_simhits <<" (may be more than one per rechit)"
      <<"\n"<< "# valid RecHits   = " <<n_valid_hits  <<" (" <<n_tracker_valid_hits<<"/"
      <<n_dt_valid_hits<<"/"<<n_csc_valid_hits<<"/"<<n_rpc_valid_hits<<" in Tracker/DT/CSC/RPC)"
      <<"\n"<< "# selected RecHits = " <<n_selected_hits  <<" (" <<n_tracker_selected_hits<<"/"
      <<n_dt_selected_hits<<"/"<<n_csc_selected_hits<<"/"<<n_rpc_selected_hits<<" in Tracker/DT/CSC/RPC)"
      <<"\n"<< "# matched RecHits = " <<n_matched_hits<<" ("<<n_tracker_matched_hits<<"/"
      <<n_dt_matched_hits<<"/"<<n_csc_matched_hits<<"/"<<n_rpc_matched_hits<<" in Tracker/DT/CSC/RPC)";
    
    if (!n_matching_simhits) {
      edm::LogWarning("MuonAssociatorByHits")
	<<"*** WARNING: no matching PSimHit found for this reco::Track !";
    }
    
    if (n_valid_hits != (*track)->found()) {
      edm::LogWarning("MuonAssociatorByHits")
	<<"*** WARNING: Number of valid RecHits in this track should be:  track->found() = "<<(*track)->found()
	<<", but getMatchedIds finds:  n_valid_hits = "<<n_valid_hits;
    }

    std::vector<SimHitIdpr> tracker_idcachev, muon_idcachev;
    
    if(n_matching_simhits) {
      int tpindex =0;
      for (TrackingParticleCollection::const_iterator trpart = tPC.begin(); trpart != tPC.end(); ++trpart, ++tpindex) {

	tracker_idcachev.clear();
	muon_idcachev.clear();

	int n_tracker_simhits = 0;
	int n_tracker_recounted_simhits = 0; 
	int n_muon_simhits = 0; 
	int n_global_simhits = 0; 
	std::vector<PSimHit> tphits;

	int n_tracker_selected_simhits = 0;
	int n_muon_selected_simhits = 0; 
	int n_global_selected_simhits = 0; 

	// shared hits are counted over the selected ones
	tracker_nshared = getShared(tracker_matchedIds, tracker_idcachev, trpart);
	muon_nshared = getShared(muon_matchedIds, muon_idcachev, trpart);
        global_nshared = tracker_nshared + muon_nshared;

        if (global_nshared == 0) continue; // if this TP shares no hits with the current reco::Track loop over 

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
		if (LayerFromDetid(dId)==LayerFromDetid(dIdOK) &&
		    dId.subdetId()==dIdOK.subdetId()) newhit = false;
	      //no grouped, splitting
	      if (!UseGrouped && UseSplitting)
		if (LayerFromDetid(dId)==LayerFromDetid(dIdOK) &&
		    dId.subdetId()==dIdOK.subdetId() &&
		    (stripDetId==0 || stripDetId->partnerDetId()!=dIdOK.rawId()))
		  newhit = false;
	      //grouped, no splitting
	      if (UseGrouped && !UseSplitting)
		if (LayerFromDetid(dId)==LayerFromDetid(dIdOK) &&
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
	  }
	}
	
	n_tracker_recounted_simhits = tphits.size();
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

	if (n_global_selected_simhits!=0) 
	  global_quality = static_cast<double>(global_nshared)/static_cast<double>(n_global_selected_simhits);
	else global_quality = 0;

	bool matchOk = true;
	if (n_selected_hits==0) matchOk = false;
	else {
	  if (n_tracker_selected_hits != 0) {
	    if (tracker_quality < tracker_quality_cut) matchOk = false;
	    
	    tracker_purity = static_cast<double>(tracker_nshared)/static_cast<double>(n_tracker_selected_hits);
	    if ((!AbsoluteNumberOfHits_track) && tracker_purity < PurityCut_track) matchOk = false;
	    
	    //if a track has just 3 hits in the Tracker we require that all 3 hits are shared
	    if (ThreeHitTracksAreSpecial && n_tracker_selected_hits==3 && tracker_nshared<3) matchOk = false;
	  }
	  
	  if (n_muon_selected_hits != 0) {
	    if (muon_quality < muon_quality_cut) matchOk = false;
	    
	    muon_purity = static_cast<double>(muon_nshared)/static_cast<double>(n_muon_selected_hits);
	    if ((!AbsoluteNumberOfHits_muon) && muon_purity < PurityCut_muon) matchOk = false;
	  }
	  
	  global_purity = static_cast<double>(global_nshared)/static_cast<double>(n_selected_hits);
	}
	//
	//edm::LogVerbatim("MuonAssociatorByHits") <<"matchOk = "<<matchOk<<", global_quality = "<<global_quality;
	
	if (matchOk) {
	  
	  outputCollection.insert(edm::Ref<TrackingParticleCollection>(TPCollectionH, tpindex), 
				  std::make_pair(tC[tindex],global_quality));
	  any_trackingParticle_matched = true;
	  
	  edm::LogVerbatim("MuonAssociatorByHits")
	    <<"************************************************************************************************************************"  
	    <<"\n"<< "TrackingParticle " << tpindex <<", q = "<<(*trpart).charge()<<", p = "<<(*trpart).p()
	    <<", pT = "<<(*trpart).pt()<<", eta = "<<(*trpart).eta()<<", phi = "<<(*trpart).phi()
	    <<"\n"<<" pdg code = "<<(*trpart).pdgId()
	    <<", made of "<<(*trpart).trackPSimHit().size()<<" PSimHits, recounted "<<n_global_simhits<<" PSimHits"
	    <<" (tracker:"<<n_tracker_recounted_simhits<<"/muons:"<<n_muon_simhits<<")"
	    <<", from "<<(*trpart).g4Tracks().size()<<" SimTrack:";
	  for(TrackingParticle::g4t_iterator g4T=(*trpart).g4Track_begin(); 
	      g4T!=(*trpart).g4Track_end(); 
	      ++g4T) {
	    edm::LogVerbatim("MuonAssociatorByHits")
	      <<" Id:"<<(*g4T).trackId()<<"/Evt:("<<(*g4T).eventId().event()<<","<<(*g4T).eventId().bunchCrossing()<<")";
	  }
	  edm::LogVerbatim("MuonAssociatorByHits")
	    <<"\t selected "<<n_global_selected_simhits<<" PSimHits"
	    <<" (tracker:"<<n_tracker_selected_simhits<<"/muons:"<<n_muon_selected_simhits<<")"
	    << "\n\t **MATCHED** with quality = "<<global_quality<<" (tracker: "<<tracker_quality<<" / muon: "<<muon_quality<<")"
	    << "\n\t               and purity = "<<global_purity<<" (tracker: "<<tracker_purity<<" / muon: "<<muon_purity<<") to:"
	    <<"\n" <<"reco::Track "<<tindex<<", q = "<<(*track)->charge()<<", p = " << (*track)->p()<<", pT = " << (*track)->pt()
	    <<", eta = "<<(*track)->momentum().eta()<<", phi = "<<(*track)->momentum().phi()
	    <<"\n"<< " made of "<<n_valid_hits<<" valid RecHits (tracker:"<<n_tracker_valid_hits<<"/muons:"<<n_muon_valid_hits<<")"
	    <<"\n"<< " selected "<<n_selected_hits<<" RecHits (tracker:"<<n_tracker_selected_hits<<"/muons:"<<n_muon_selected_hits<<")";
	}
	else {
	  // print something only if this TrackingParticle shares some hits with the current reco::Track
	  if (global_nshared >0) {
	    if (printRtS) edm::LogVerbatim("MuonAssociatorByHits")
	      <<"************************************************************************************************************************"  
	      <<"\n"<<"TrackingParticle " << tpindex <<", q = "<<(*trpart).charge()<<", p = "<<(*trpart).p()
	      <<", pT = "<<(*trpart).pt()<<", eta = "<<(*trpart).eta()<<", phi = "<<(*trpart).phi()
	      <<"\n"<<" pdg code = "<<(*trpart).pdgId()
	      <<", made of "<<(*trpart).trackPSimHit().size()<<" PSimHits, recounted "<<n_global_simhits<<" PSimHits"
	      <<" (tracker:"<<n_tracker_recounted_simhits<<"/muons:"<<n_muon_simhits<<")"
	      <<", from "<<(*trpart).g4Tracks().size()<<" SimTrack:";
	    for(TrackingParticle::g4t_iterator g4T=(*trpart).g4Track_begin(); 
		g4T!=(*trpart).g4Track_end(); 
		++g4T) {
	      if (printRtS) edm::LogVerbatim("MuonAssociatorByHits")
		<<" Id:"<<(*g4T).trackId()<<"/Evt:("<<(*g4T).eventId().event()<<","<<(*g4T).eventId().bunchCrossing()<<")";
	    }
	    if (printRtS) edm::LogVerbatim("MuonAssociatorByHits")
	      <<"\t selected "<<n_global_selected_simhits<<" PSimHits"
	      <<" (tracker:"<<n_tracker_selected_simhits<<"/muons:"<<n_muon_selected_simhits<<")"
	      <<"\n\t NOT matched  to reco::Track "<<tindex
	      <<" with quality = "<<global_quality<<" (tracker: "<<tracker_quality<<" / muon: "<<muon_quality<<")"
	      <<"\n\t and purity = "<<global_purity<<" (tracker: "<<tracker_purity<<" / muon: "<<muon_purity<<")";
	  }
	}
      }  // loop over TrackingParticle's
    }   // if(n_matching_simhits)
  }    // loop over reco Tracks
  
  if (!any_trackingParticle_matched) {
    edm::LogVerbatim("MuonAssociatorByHits")
      <<"\n"
      <<"************************************************************************************************************************"
      << "\n NO TrackingParticle associated to ANY input reco::Track ! \n"
      <<"************************************************************************************************************************"<<"\n";  
  } else {
    edm::LogVerbatim("MuonAssociatorByHits")
      <<"************************************************************************************************************************"<<"\n";  
  }
  
  delete trackertruth;
  outputCollection.post_insert();
  return outputCollection;
}

int MuonAssociatorByHits::LayerFromDetid(const DetId& detId ) const
{
  int layerNumber=0;
  if (detId.det() != DetId::Tracker) return layerNumber;

  unsigned int subdetId = static_cast<unsigned int>(detId.subdetId()); 
  if ( subdetId == StripSubdetector::TIB) 
    { 
      TIBDetId tibid(detId.rawId()); 
      layerNumber = tibid.layer();
    }
  else if ( subdetId ==  StripSubdetector::TOB )
    { 
      TOBDetId tobid(detId.rawId()); 
      layerNumber = tobid.layer();
    }
  else if ( subdetId ==  StripSubdetector::TID) 
    { 
      TIDDetId tidid(detId.rawId());
      layerNumber = tidid.wheel();
    }
  else if ( subdetId ==  StripSubdetector::TEC )
    { 
      TECDetId tecid(detId.rawId()); 
      layerNumber = tecid.wheel(); 
    }
  else if ( subdetId ==  PixelSubdetector::PixelBarrel ) 
    { 
      PXBDetId pxbid(detId.rawId()); 
      layerNumber = pxbid.layer();  
    }
  else if ( subdetId ==  PixelSubdetector::PixelEndcap ) 
    { 
      PXFDetId pxfid(detId.rawId()); 
      layerNumber = pxfid.disk();  
    }
  else
    edm::LogWarning("MuonAssociatorByHits") << "Unknown Tracker subdetector: subdetId = " <<  subdetId;
  
  return layerNumber;
} 

template<typename iter>
void MuonAssociatorByHits::getMatchedIds(std::vector<SimHitIdpr>& tracker_matchedIds,std::vector<SimHitIdpr>& muon_matchedIds,
					 int& n_valid_hits, int& n_tracker_valid_hits, 
					 int& n_dt_valid_hits, int& n_csc_valid_hits, int& n_rpc_valid_hits,
					 int& n_selected_hits,int& n_tracker_selected_hits,
					 int& n_dt_selected_hits,int& n_csc_selected_hits,int& n_rpc_selected_hits,
					 int& n_matched_hits, int& n_tracker_matched_hits, 
					 int& n_dt_matched_hits, int& n_csc_matched_hits, int& n_rpc_matched_hits,
					 iter begin,
					 iter end,
					 TrackerHitAssociator* trackertruth, 
					 DTHitAssociator & dttruth,
					 MuonTruth & csctruth,
					 RPCHitAssociator & rpctruth,
					 bool printRtS) const {
  tracker_matchedIds.clear();
  muon_matchedIds.clear();
  
  n_valid_hits = 0;            // number of valid rechits (Total)
  n_tracker_valid_hits = 0;    //                         (Tracker)
  n_dt_valid_hits = 0;         //                         (DT)
  n_csc_valid_hits = 0;        //                         (CSC)
  n_rpc_valid_hits = 0;        //                         (RPC)

  n_selected_hits = 0;          // number of selected hits   (Total)
  n_tracker_selected_hits = 0;  //                           (Tracker)
  n_dt_selected_hits = 0;       //                           (DT)
  n_csc_selected_hits = 0;      //                           (CSC)
  n_rpc_selected_hits = 0;      //                           (RPC)

  n_matched_hits = 0;          // number of associated rechits (Total)
  n_tracker_matched_hits = 0;  //                              (Tracker)
  n_dt_matched_hits = 0;       //                              (DT)
  n_csc_matched_hits = 0;      //                              (CSC)
  n_rpc_matched_hits = 0;      //                              (RPC)
  
  std::vector<SimHitIdpr> SimTrackIds;

  int iloop = 0;
  for (iter it = begin;  it != end; it++, iloop++) {
    stringstream hit_index;     
    hit_index<<iloop;

    unsigned int detid = getHitPtr(it)->geographicalId().rawId();    
    stringstream detector_id;
    detector_id<<detid;

    string hitlog = "TrackingRecHit "+hit_index.str();
    string wireidlog;
    std::vector<string> DTSimHits;
    
    DetId::Detector det = getHitPtr(it)->geographicalId().det();
    int subdet = getHitPtr(it)->geographicalId().subdetId();
    
    if (getHitPtr(it)->isValid()) {
      n_valid_hits++;
      SimTrackIds.clear();	  
      
      if (det == DetId::Tracker) {
	hitlog = hitlog+" -Tracker - detID = "+detector_id.str();
	
	n_tracker_valid_hits++;
	SimTrackIds = trackertruth->associateHitId(*getHitPtr(it));
	
	if (UseTracker) {
	  n_tracker_selected_hits++;
	  n_selected_hits++;
	  
	  if(!SimTrackIds.empty()) {
	    n_tracker_matched_hits++;
	    n_matched_hits++;
	    for(size_t j=0; j<SimTrackIds.size(); j++){
	      tracker_matchedIds.push_back(SimTrackIds[j]);			
	    }
	  }
	}
      }
      else if (det == DetId::Muon) {

	if (subdet == MuonSubdetId::DT) {
	  hitlog = hitlog+" -Muon DT - detID = "+detector_id.str();	  

	  n_dt_valid_hits++;
	  SimTrackIds = dttruth.associateHitId(*getHitPtr(it));  
	  
	  if (UseMuon) {
	    n_dt_selected_hits++;
	    n_selected_hits++;

	    if (!SimTrackIds.empty()) {
	      n_dt_matched_hits++;
	      n_matched_hits++;
	      for(unsigned int j=0; j<SimTrackIds.size(); j++) {
		muon_matchedIds.push_back(SimTrackIds[j]);
	      }
	    }
	  }
	  if (dumpDT) {
	    const DTRecHit1D * dtrechit = dynamic_cast<const DTRecHit1D *>(&(*getHitPtr(it)));
	    DTWireId wireid = dtrechit->wireId();
	    stringstream wid; 
	    wid<<wireid;
	    std::vector<PSimHit> dtSimHits = dttruth.associateHit(*getHitPtr(it));
	    
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
	  } 
	}
	
	else if (subdet == MuonSubdetId::CSC) {
	  hitlog = hitlog+" -Muon CSC- detID = "+detector_id.str();	  
	  
	  n_csc_valid_hits++;
	  SimTrackIds = csctruth.associateHitId(*getHitPtr(it)); 

	  if (UseMuon) {
	    n_csc_selected_hits++;
	    n_selected_hits++;
	    
	    if (!SimTrackIds.empty()) {
	      n_csc_matched_hits++;
	      n_matched_hits++;
	      for(unsigned int j=0; j<SimTrackIds.size(); j++) {
		muon_matchedIds.push_back(SimTrackIds[j]);
	      }
	    }
	  } 
	}
	
	else if (subdet == MuonSubdetId::RPC) {
	  hitlog = hitlog+" -Muon RPC- detID = "+detector_id.str();	  
	  
	  n_rpc_valid_hits++;
	  SimTrackIds = rpctruth.associateRecHit(*getHitPtr(it));  
	  
	  if (UseMuon) {
	    n_rpc_selected_hits++;
	    n_selected_hits++;
	    
	    if (!SimTrackIds.empty()) {
	      n_rpc_matched_hits++;
	      n_matched_hits++;
	      for(unsigned int j=0; j<SimTrackIds.size(); j++) {
		muon_matchedIds.push_back(SimTrackIds[j]);
	      }
	    }
	  } 
	}
	
      } else if (printRtS) edm::LogVerbatim("MuonAssociatorByHits")
	<<"TrackingRecHit "<<iloop<<"  *** WARNING *** Unexpected Hit from Detector = "<<det;
      
      hitlog = hitlog + write_matched_simtracks(SimTrackIds);

      if (printRtS) edm::LogVerbatim("MuonAssociatorByHits") << hitlog;
      if (printRtS && dumpDT && det==DetId::Muon && subdet==MuonSubdetId::DT) {
	edm::LogVerbatim("MuonAssociatorByHits") <<wireidlog;
	for (unsigned int j=0; j<DTSimHits.size(); j++) {
	  edm::LogVerbatim("MuonAssociatorByHits") <<DTSimHits[j];
	}
      }
      
    } else if (printRtS) edm::LogVerbatim("MuonAssociatorByHits")
      <<"TrackingRecHit "<<iloop<<"  *** WARNING *** Invalid Hit On DetId.rawId() : "<<detid;
    
  }//trackingRecHit loop
}

int MuonAssociatorByHits::getShared(std::vector<SimHitIdpr>& matchedIds, 
				    std::vector<SimHitIdpr>& idcachev,
				    TrackingParticleCollection::const_iterator trpart) const {
  int nshared = 0;

  for(size_t j=0; j<matchedIds.size(); j++) {
    if(find(idcachev.begin(), idcachev.end(),matchedIds[j]) == idcachev.end() ) {
      idcachev.push_back(matchedIds[j]);
      
      for (TrackingParticle::g4t_iterator simtrack = trpart->g4Track_begin(); simtrack !=  trpart->g4Track_end(); ++simtrack) {
	if((*simtrack).trackId() == matchedIds[j].first && trpart->eventId() == matchedIds[j].second) {
	  
	  int countedhits = std::count(matchedIds.begin(), matchedIds.end(), matchedIds[j]);
	  nshared += countedhits;
	}	
      } 
    }
  }
  return nshared;
}

std::string MuonAssociatorByHits::write_matched_simtracks(const std::vector<SimHitIdpr>& SimTrackIds) const {

  string hitlog;

  if (!SimTrackIds.empty()) {
    hitlog = " matched to SimTrack";
  
    for(size_t j=0; j<SimTrackIds.size(); j++){
      stringstream trackid;  
      trackid<<SimTrackIds[j].first;
      
      stringstream evtid;    
      evtid<<SimTrackIds[j].second.event();
      
      stringstream bunchxid; 
      bunchxid<<SimTrackIds[j].second.bunchCrossing();
      
      hitlog = hitlog+" Id:"+trackid.str()+"/Evt:("+evtid.str()+","+bunchxid.str()+") ";
    }
  } else hitlog = "  *** UNMATCHED ***";

  return hitlog;
}
