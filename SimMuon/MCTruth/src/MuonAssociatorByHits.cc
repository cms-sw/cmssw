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

using namespace reco;
using namespace std;

MuonAssociatorByHits::MuonAssociatorByHits (const edm::ParameterSet& conf) :  
  AbsoluteNumberOfHits(conf.getParameter<bool>("AbsoluteNumberOfHits")),
  SimToRecoDenominator(conf.getParameter<string>("SimToRecoDenominator")),
  theMinHitCut(conf.getParameter<double>("MinHitCut")),
  UsePixels(conf.getParameter<bool>("UsePixels")),
  UseGrouped(conf.getParameter<bool>("UseGrouped")),
  UseSplitting(conf.getParameter<bool>("UseSplitting")),
  ThreeHitTracksAreSpecial(conf.getParameter<bool>("ThreeHitTracksAreSpecial")),
  debug(conf.getParameter<bool>("debug")),
  crossingframe(conf.getParameter<bool>("crossingframe")),
  conf_(conf)
{
  LogTrace("MuonAssociatorByHits") << "constructing  MuonAssociatorByHits" << conf_.dump();
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

  float tracker_quality=0;
  float muon_quality=0;
  float global_quality=0;

  std::vector< SimHitIdpr> tracker_matchedIds, muon_matchedIds;

  RecoToSimCollection  outputCollection;

  // Tracker hit association  
  TrackerHitAssociator * trackertruth = new TrackerHitAssociator::TrackerHitAssociator(*e, conf_);
  // CSC hit association
  MuonTruth csctruth;
  csctruth.eventSetup(*e);
  // DT hit association
  DTHitAssociator dttruth(*e,*setup,conf_);
  // RPC hit association
  RPCHitAssociator rpctruth(*e,*setup,conf_);

  const TrackingParticleCollection tPC   = *(TPCollectionH.product());
  
  if (debug) {
    // reco::Track collection
    edm::LogVerbatim("MuonAssociatorByHits")<<"reco::Track collection --- size = "<<tC.size();
    int i = 0;
    for (edm::RefToBaseVector<reco::Track>::const_iterator ITER=tC.begin(); ITER!=tC.end(); ITER++, i++) {
      edm::LogVerbatim("MuonAssociatorByHits")
	<<"Track "<<i<<" : p = "<<(*ITER)->p()<<", charge = "<<(*ITER)->charge()
	<<", pT = "<<(*ITER)->pt()<<", eta = "<<(*ITER)->eta()<<", phi = "<<(*ITER)->phi();    
    }

    // TrackingParticle collection
    edm::LogVerbatim("MuonAssociatorByHits")<<"TrackingParticle collection --- size = "<<tPC.size();
    int j = 0;
    for(TrackingParticleCollection::const_iterator ITER=tPC.begin(); ITER!=tPC.end(); ITER++, j++) {
      edm::LogVerbatim("MuonAssociatorByHits")
	<<"TrackingParticle "<<j<<" made of "<<ITER->trackPSimHit().size()<<" PSimHit"
	<<" (in "<<ITER->matchedHit()<<" layers)"
	<<" from "<<ITER->g4Tracks().size()<<" SimTrack:";
      for (TrackingParticle::g4t_iterator g4T=ITER->g4Track_begin(); g4T!=ITER->g4Track_end(); g4T++) {
	edm::LogVerbatim("MuonAssociatorByHits")
	  <<" Id:"<<g4T->trackId()<<"/Evt:("<<g4T->eventId().event()<<","<<g4T->eventId().bunchCrossing()<<")";
      }    
      edm::LogVerbatim("MuonAssociatorByHits")
	<<" pdgId = "<<ITER->pdgId()<<", p = "<<ITER->p()
	<<", pT = "<<ITER->pt()<<", eta = "<<ITER->eta()<<", phi = "<<ITER->phi();
    }

    // SimTrack collection
    edm::Handle<CrossingFrame<SimTrack> > cf_simtracks;
    edm::Handle<edm::SimTrackContainer> simTrackCollection;
    
    if (crossingframe) {
      e->getByLabel("mix",cf_simtracks);
      auto_ptr<MixCollection<SimTrack> > SimTk( new MixCollection<SimTrack>(cf_simtracks.product()) );
      edm::LogVerbatim("MuonAssociatorByHits")<<"CrossingFrame<SimTrack> collection --- size = "<<SimTk->size();
      if ((int)(SimTk->size()) != (int)(tPC.size())) {
	edm::LogWarning("MuonAssociatorByHits")
	  <<"WARNING : Nr.SimTracks = "<<SimTk->size()<<", Nr.TrackingParticles = "<<tPC.size();
      }
      int k = 0;
      for (MixCollection<SimTrack>::MixItr ITER=SimTk->begin(); ITER!=SimTk->end(); ITER++, k++) {
	edm::LogVerbatim("MuonAssociatorByHits")
	  <<"SimTrack "<<k
	  <<" - Id:"<<ITER->trackId()<<"/Evt:("<<ITER->eventId().event()<<","<<ITER->eventId().bunchCrossing()<<")"
	  <<" pdgId = "<<ITER->type()<<", p = "<<ITER->momentum().P()<<", pT = "<<ITER->momentum().Pt()
	  <<", eta = "<<ITER->momentum().Eta()<<", phi = "<<ITER->momentum().Phi();
      }
    }
    else {
      e->getByLabel("g4SimHits",simTrackCollection);
      const edm::SimTrackContainer simTC = *(simTrackCollection.product());
      edm::LogVerbatim("MuonAssociatorByHits")<<"SimTrack collection --- size = "<<simTC.size()<<endl;
      if (simTC.size() != tPC.size()) {
	edm::LogWarning("MuonAssociatorByHits")
	  <<"WARNING : Nr.SimTracks = "<<simTC.size()<<", Nr.TrackingParticles = "<<tPC.size();
      }
      int k = 0;
      for(edm::SimTrackContainer::const_iterator ITER=simTC.begin(); ITER!=simTC.end(); ITER++, k++){
	edm::LogVerbatim("MuonAssociatorByHits")
	  <<"SimTrack "<<k
	  <<" - Id:"<<ITER->trackId()<<"/Evt:("<<ITER->eventId().event()<<","<<ITER->eventId().bunchCrossing()<<")"
	  <<" pdgId = "<<ITER->type()<<", p = "<<ITER->momentum().P()<<", pT = "<<ITER->momentum().Pt()
	  <<", eta = "<<ITER->momentum().Eta()<<", phi = "<<ITER->momentum().Phi();
      }
    }
  }
  
  int tindex=0;
  for (edm::RefToBaseVector<reco::Track>::const_iterator track=tC.begin(); track!=tC.end(); track++, tindex++) {
    edm::LogVerbatim("MuonAssociatorByHits")
      << "reco::Track "<<tindex
      <<", pT = " << (*track)->pt()<<", eta = "<<(*track)->momentum().eta()<<", phi = "<<(*track)->momentum().phi()
      <<", number of RecHits = "<<(*track)->recHitsSize()<<" ("<<(*track)->found()<<" valid hits, "<<(*track)->lost()<<" lost hits)"<<endl;
    
    tracker_matchedIds.clear();
    muon_matchedIds.clear();
    int n_matching_simhits = 0;

    int n_valid_hits = 0;          // number of valid hits      (Total)
    int n_tracker_valid_hits = 0;  //                           (Tracker)
    int n_muon_valid_hits = 0;     //                           (DT+CSC+RPC)
    int n_dt_valid_hits = 0;       //                           (DT)
    int n_csc_valid_hits = 0;      //                           (CSC)
    int n_rpc_valid_hits = 0;      //                           (RPC)

    int n_matched_hits = 0;          // number of matched hits    (Total)
    int n_tracker_matched_hits = 0;  //                           (Tracker)
    int n_muon_matched_hits = 0;     //                           (DT+CSC+RPC)
    int n_dt_matched_hits = 0;       //                           (DT)
    int n_csc_matched_hits = 0;      //                           (CSC)
    int n_rpc_matched_hits = 0;      //                           (RPC)

    getMatchedIds<trackingRecHit_iterator>(tracker_matchedIds, muon_matchedIds,
					   n_valid_hits, n_tracker_valid_hits, n_dt_valid_hits, n_csc_valid_hits, n_rpc_valid_hits,
					   n_matched_hits, n_tracker_matched_hits, n_dt_matched_hits, n_csc_matched_hits, n_rpc_matched_hits,
					   (*track)->recHitsBegin(), (*track)->recHitsEnd(), 
					   trackertruth, dttruth, csctruth, rpctruth);

    n_matching_simhits = tracker_matchedIds.size() + muon_matchedIds.size(); 
    n_muon_valid_hits = n_dt_valid_hits + n_csc_valid_hits + n_rpc_valid_hits;
    n_muon_matched_hits = n_dt_matched_hits + n_csc_matched_hits + n_rpc_matched_hits;

    edm::LogVerbatim("MuonAssociatorByHits")
      <<"\n"<<"*** # all RecHits = "<<(*track)->recHitsSize()
      <<", # matching SimHits = " << n_matching_simhits <<" (may be more than one per rechit)";

    edm::LogVerbatim("MuonAssociatorByHits")
      <<"# valid RecHits   = " <<n_valid_hits  <<" (" <<n_tracker_valid_hits<<"/"
      <<n_dt_valid_hits<<"/"<<n_csc_valid_hits<<"/"<<n_rpc_valid_hits<<" in Tracker/DT/CSC/RPC)";  
    
    edm::LogVerbatim("MuonAssociatorByHits")
      <<"# matched RecHits = " <<n_matched_hits<<" ("<<n_tracker_matched_hits<<"/"
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
	<<"\n"<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";
      edm::LogVerbatim("MuonAssociatorByHits")
	<<"reco::Track "<<tindex<<", q = "<<(*track)->charge()<<", p = " << (*track)->p()<<", pT = " << (*track)->pt()
	<<", eta = "<<(*track)->momentum().eta()<<", phi = "<<(*track)->momentum().phi();

      int tpindex = 0;
      //      for (TrackingParticleCollection::const_iterator t = tPC.begin(); t != tPC.end(); ++t, ++tpindex) {
      for (TrackingParticleCollection::const_iterator trpart = tPC.begin(); trpart != tPC.end(); ++trpart, ++tpindex) {
	tracker_idcachev.clear();
	tracker_nshared = getShared(tracker_matchedIds, tracker_idcachev, trpart);
	muon_idcachev.clear();
	muon_nshared = getShared(muon_matchedIds, muon_idcachev, trpart);
        global_nshared = tracker_nshared + muon_nshared;

	if (AbsoluteNumberOfHits) tracker_quality = static_cast<double>(tracker_nshared);
	else if(n_tracker_valid_hits!=0) tracker_quality = (static_cast<double>(tracker_nshared)/static_cast<double>(n_tracker_valid_hits));
	else tracker_quality = 0;

	int n_muon_hits = n_dt_valid_hits + n_csc_valid_hits + n_rpc_valid_hits;
	if (AbsoluteNumberOfHits) muon_quality = static_cast<double>(muon_nshared);
	else if(n_muon_hits!=0) muon_quality = (static_cast<double>(muon_nshared)/static_cast<double>(n_muon_hits));
	else muon_quality = 0;

	int n_tot_hits = n_tracker_valid_hits + n_muon_hits;
	if (AbsoluteNumberOfHits) global_quality = static_cast<double>(global_nshared);
	else if(n_tot_hits!=0) global_quality = (static_cast<double>(global_nshared)/static_cast<double>(n_tot_hits));
	else global_quality = 0;

	bool matchOk = true;
	if (n_tracker_valid_hits != 0) {
	  if (tracker_quality < theMinHitCut) matchOk = false;
	}
	if (n_muon_hits != 0) {
	  if (muon_quality < theMinHitCut) matchOk = false;
	}
  
	if (matchOk) {
	  //if a track has just 3 hits we require that all 3 hits are shared
	  if (ThreeHitTracksAreSpecial && n_tracker_valid_hits==3 && tracker_nshared<3) continue;

	  if(!AbsoluteNumberOfHits && tracker_quality>1.) 
	    edm::LogWarning("MuonAssociatorByHits")
	      <<" **** fraction > 1 "
	      <<", tracker_nshared = "<<tracker_nshared<<", n_tracker_valid_hits = "<<n_tracker_valid_hits;
	  if(!AbsoluteNumberOfHits && muon_quality>1.) 
	    edm::LogWarning("MuonAssociatorByHits")
	      <<" **** fraction > 1 " 
	      <<", muon_nshared = "<<muon_nshared<<", n_muon_hits = "<<n_muon_hits;
	  
	  outputCollection.insert(tC[tindex],
				  std::make_pair(edm::Ref<TrackingParticleCollection>(TPCollectionH, tpindex),
						 global_quality));
	  edm::LogVerbatim("MuonAssociatorByHits")<<" **MATCHED** to TrackingParticle " <<tpindex 
	      << " with quality = "<<global_quality<<" (tracker: "<<tracker_quality<<" / muon: "<<muon_quality<<")";
	  edm::LogVerbatim("MuonAssociatorByHits")<<"    (pdgId = "<<(*trpart).pdgId()<<", "<<(*trpart).g4Tracks().size()<<" SimTrack)"
	      <<", p = "<<(*trpart).p()<<", pT = "<<(*trpart).pt()<<", eta = "<<(*trpart).eta()<<", phi = "<<(*trpart).phi();
	}
	else {
	  edm::LogVerbatim("MuonAssociatorByHits")<<" NOT matched to TrackingParticle "<<tpindex
	      << " with quality = "<<global_quality<<" (tracker: "<<tracker_quality<<" / muon: "<<muon_quality<<")";
	}
	
      }    //  loop over TrackingParticle
      edm::LogVerbatim("MuonAssociatorByHits")
	<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<endl;
      
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

  float tracker_quality=0;
  float muon_quality=0;
  float global_quality=0;

  std::vector< SimHitIdpr> tracker_matchedIds, muon_matchedIds;

  SimToRecoCollection  outputCollection;

  // Tracker hit association  
  TrackerHitAssociator * trackertruth = new TrackerHitAssociator::TrackerHitAssociator(*e, conf_);
  // CSC hit association
  MuonTruth csctruth;
  csctruth.eventSetup(*e);
  // DT hit association
  DTHitAssociator dttruth(*e,*setup,conf_);  
  // RPC hit association
  RPCHitAssociator rpctruth(*e,*setup,conf_);
  
  const TrackingParticleCollection tPC   = *(TPCollectionH.product());

  int tindex=0;
  for (edm::RefToBaseVector<reco::Track>::const_iterator track=tC.begin(); track!=tC.end(); track++, tindex++) {
    edm::LogVerbatim("MuonAssociatorByHits")
      << "reco::Track "<<tindex
      <<", pT = " << (*track)->pt()<<", eta = "<<(*track)->momentum().eta()<<", phi = "<<(*track)->momentum().phi()
      <<", number of RecHits = "<<(*track)->recHitsSize()<<" ("<<(*track)->found()<<" valid hits, "<<(*track)->lost()<<" lost hits)"<<endl;
    
    tracker_matchedIds.clear();
    muon_matchedIds.clear();
    int n_matching_simhits = 0;

    int n_valid_hits = 0;          // number of valid hits      (Total)
    int n_tracker_valid_hits = 0;  //                           (Tracker)
    int n_muon_valid_hits = 0;     //                           (DT+CSC+RPC)
    int n_dt_valid_hits = 0;       //                           (DT)
    int n_csc_valid_hits = 0;      //                           (CSC)
    int n_rpc_valid_hits = 0;      //                           (RPC)

    int n_matched_hits = 0;          // number of matched hits    (Total)
    int n_tracker_matched_hits = 0;  //                           (Tracker)
    int n_muon_matched_hits = 0;     //                           (DT+CSC+RPC)
    int n_dt_matched_hits = 0;       //                           (DT)
    int n_csc_matched_hits = 0;      //                           (CSC)
    int n_rpc_matched_hits = 0;      //                           (RPC)

    getMatchedIds<trackingRecHit_iterator>(tracker_matchedIds, muon_matchedIds,
					   n_valid_hits, n_tracker_valid_hits, n_dt_valid_hits, n_csc_valid_hits, n_rpc_valid_hits,
					   n_matched_hits, n_tracker_matched_hits, n_dt_matched_hits, n_csc_matched_hits, n_rpc_matched_hits,
					   (*track)->recHitsBegin(), (*track)->recHitsEnd(), 
					   trackertruth, dttruth, csctruth, rpctruth);

    n_matching_simhits = tracker_matchedIds.size() + muon_matchedIds.size(); 
    n_muon_valid_hits = n_dt_valid_hits + n_csc_valid_hits + n_rpc_valid_hits;
    n_muon_matched_hits = n_dt_matched_hits + n_csc_matched_hits + n_rpc_matched_hits;

    edm::LogVerbatim("MuonAssociatorByHits")
      <<"\n"<<"*** # all RecHits = "<<(*track)->recHitsSize()
      <<", # matching PSimHit = " << n_matching_simhits <<" (may be more than one per rechit)";
    
    edm::LogVerbatim("MuonAssociatorByHits")
      <<"# valid RecHits   = " <<n_valid_hits  <<" (" <<n_tracker_valid_hits<<"/"
      <<n_dt_valid_hits<<"/"<<n_csc_valid_hits<<"/"<<n_rpc_valid_hits<<" in Tracker/DT/CSC/RPC)";  
    
    edm::LogVerbatim("MuonAssociatorByHits")
      <<"# matched RecHits = " <<n_matched_hits<<" ("<<n_tracker_matched_hits<<"/"
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
	//	LogTrace("MuonAssociatorByHits") << "NEW TrackingParticle "<<tpindex;
	//	LogTrace("MuonAssociatorByHits") << "number of simhits for this TP: "  << trpart->trackPSimHit().size();
	tracker_idcachev.clear();
	muon_idcachev.clear();

	int n_tracker_simhits = 0;
	int n_tracker_recounted_simhits = 0; 
	int n_muon_simhits = 0; 
	int n_global_simhits = 0; 
	std::vector<PSimHit> tphits;

	tracker_nshared = getShared(tracker_matchedIds, tracker_idcachev, trpart);
	muon_nshared = getShared(muon_matchedIds, muon_idcachev, trpart);
        global_nshared = tracker_nshared + muon_nshared;

	//	LogTrace("MuonAssociatorByHits") << "recounting of Tracker simhits";
	for(std::vector<PSimHit>::const_iterator TPhit = trpart->pSimHit_begin(); TPhit != trpart->pSimHit_end(); TPhit++) {
          DetId dId = DetId(TPhit->detUnitId());
	  DetId::Detector detector = dId.det();
	  //	  LogTrace("MuonAssociatorByHits") << "detector = "<<dId.det()<<endl;
	  
	  if (detector == DetId::Tracker) {
	    n_tracker_simhits++;

	    unsigned int subdetId = static_cast<unsigned int>(dId.subdetId());
	    if (!UsePixels && (subdetId==PixelSubdetector::PixelBarrel || subdetId==PixelSubdetector::PixelEndcap) )
	      continue;

	    SiStripDetId* stripDetId = 0;
	    if (subdetId==SiStripDetId::TIB||subdetId==SiStripDetId::TOB||
		subdetId==SiStripDetId::TID||subdetId==SiStripDetId::TEC)
	      stripDetId= new SiStripDetId(dId);
	    //	    LogTrace("MuonAssociatorByHits") << "consider hit SUBDET = " << subdetId
	    //			<< " layer = " << LayerFromDetid(dId) 
	    //			<< " id = " << dId.rawId();
	    bool newhit = true;
	    for(std::vector<PSimHit>::const_iterator TPhitOK = tphits.begin(); TPhitOK != tphits.end(); TPhitOK++) {
	      DetId dIdOK = DetId(TPhitOK->detUnitId());
	      //	      LogTrace("MuonAssociatorByHits") << "\t\tcompare with SUBDET = " << dIdOK.subdetId()
	      //			  << " layer = " << LayerFromDetid(dIdOK)
	      //			  << " id = " << dIdOK.rawId();
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
	      //	      LogTrace("MuonAssociatorByHits") << "\t\tok";
	      tphits.push_back(*TPhit);
	    }
	    //	    else LogTrace("MuonAssociatorByHits") << "\t\tno";
	    delete stripDetId;
	  }
	  else if (detector == DetId::Muon) {
	    //	    LogTrace("MuonAssociatorByHits") << "\t\tok";
	    n_muon_simhits++;
	  }
	}
	
	n_tracker_recounted_simhits = tphits.size();
	n_global_simhits = n_tracker_recounted_simhits + n_muon_simhits;
	edm::LogVerbatim("MuonAssociatorByHits") 
	  <<"\n" << "TrackingParticle #" << tpindex << " has " << trpart->trackPSimHit().size() << " simhits";
	edm::LogVerbatim("MuonAssociatorByHits") 
	  << "Tracker Final count: n simhits = " << n_tracker_simhits << ", recounted (layers) = " << n_tracker_recounted_simhits 
	  << ", nshared = " << tracker_nshared << ", nrechit = " << n_tracker_valid_hits;
	edm::LogVerbatim("MuonAssociatorByHits") 
	  << "Muon Final count: n simhits = " << n_muon_simhits 
	  << ", nshared = " << muon_nshared << ", nrechit = " << n_muon_valid_hits;

	if (AbsoluteNumberOfHits) tracker_quality = static_cast<double>(tracker_nshared);
	else if(SimToRecoDenominator == "sim" && n_tracker_recounted_simhits!=0) 
	  tracker_quality = static_cast<double>(tracker_nshared)/static_cast<double>(n_tracker_recounted_simhits);
	else if(SimToRecoDenominator == "reco" && n_tracker_valid_hits!=0) 
	  tracker_quality = static_cast<double>(tracker_nshared)/static_cast<double>(n_tracker_valid_hits);
	else tracker_quality = 0;
	
	int n_muon_hits = n_dt_valid_hits + n_csc_valid_hits + n_rpc_valid_hits;
	if (AbsoluteNumberOfHits) muon_quality = static_cast<double>(muon_nshared);
	else if(SimToRecoDenominator == "sim" && n_muon_simhits!=0) 
	  muon_quality = static_cast<double>(muon_nshared)/static_cast<double>(n_muon_simhits);
	else if(SimToRecoDenominator == "reco" && n_muon_hits!=0) 
	  muon_quality = static_cast<double>(muon_nshared)/static_cast<double>(n_muon_hits);
	else muon_quality = 0;

	if (AbsoluteNumberOfHits) global_quality = static_cast<double>(global_nshared);
	else if(SimToRecoDenominator == "sim" && n_global_simhits!=0) 
	  global_quality = static_cast<double>(global_nshared)/static_cast<double>(n_global_simhits);
	else if(SimToRecoDenominator == "reco" && n_valid_hits!=0) 
	  global_quality = static_cast<double>(global_nshared)/static_cast<double>(n_valid_hits);
	else global_quality = 0;

	bool matchOk = true;
	if (n_tracker_valid_hits != 0) {
	  if (tracker_quality < theMinHitCut) matchOk = false;
	}
	if (n_muon_hits != 0) {
	  if (muon_quality < theMinHitCut) matchOk = false;
	}
	
	edm::LogVerbatim("MuonAssociatorByHits")
	  <<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";  
	if (matchOk) {
	  outputCollection.insert(edm::Ref<TrackingParticleCollection>(TPCollectionH, tpindex), 
				  std::make_pair(tC[tindex],global_quality));
	  edm::LogVerbatim("MuonAssociatorByHits") 
	    << "TrackingParticle " << tpindex
	    <<"    (pdgId = "<<(*trpart).pdgId()<<", "<<(*trpart).g4Tracks().size()<<" SimTrack)"
	    <<", p = "<<(*trpart).p()<<", pT = "<<(*trpart).pt()<<", eta = "<<(*trpart).eta()<<", phi = "<<(*trpart).phi();
	  edm::LogVerbatim("MuonAssociatorByHits")
	    <<" associated to reco::Track "<<tindex<<", q = "<<(*track)->charge()<<", p = " << (*track)->p()<<", pT = " << (*track)->pt()
	    <<", eta = "<<(*track)->momentum().eta()<<", phi = "<<(*track)->momentum().phi();
	  edm::LogVerbatim("MuonAssociatorByHits")
	    << " with quality = " << global_quality <<" (tracker: "<<tracker_quality<<" / muon: "<<muon_quality<<")";

	}
	//if a track has just 3 hits we require that all 3 hits are shared
	else if(ThreeHitTracksAreSpecial && n_tracker_valid_hits==3) {
	  bool putIt = false;
	  if (AbsoluteNumberOfHits && tracker_quality==3){
	    putIt = true;
	  } else if(!AbsoluteNumberOfHits && tracker_nshared==3){
	    putIt = true;
	    tracker_quality = 1.;
	  }
	  if (putIt) {
	    outputCollection.insert(edm::Ref<TrackingParticleCollection>(TPCollectionH, tpindex), 
				    std::make_pair(tC[tindex],global_quality));
	    edm::LogVerbatim("MuonAssociatorByHits") 
	      << "TrackingParticle " << tpindex 
	      <<"    (pdgId = "<<(*trpart).pdgId()<<", "<<(*trpart).g4Tracks().size()<<" SimTrack)"
	      <<", p = "<<(*trpart).p()<<", pT = "<<(*trpart).pt()<<", eta = "<<(*trpart).eta()<<", phi = "<<(*trpart).phi();
	    edm::LogVerbatim("MuonAssociatorByHits")
	      <<" associated to reco::Track "<<tindex<<", q = "<<(*track)->charge()<<", p = " << (*track)->p()<<", pT = " << (*track)->pt()
	      <<", eta = "<<(*track)->momentum().eta()<<", phi = "<<(*track)->momentum().phi();
	    edm::LogVerbatim("MuonAssociatorByHits")
	      << " with quality = " << global_quality <<" (tracker: "<<tracker_quality<<" / muon: "<<muon_quality<<")";
	  }
	}
	else {
	  edm::LogVerbatim("MuonAssociatorByHits") 
	    << "TrackingParticle " << tpindex 
	    <<"    (pdgId = "<<(*trpart).pdgId()<<", "<<(*trpart).g4Tracks().size()<<" SimTrack)"
	    <<", p = "<<(*trpart).p()<<", pT = "<<(*trpart).pt()<<", eta = "<<(*trpart).eta()<<", phi = "<<(*trpart).phi();
	  edm::LogVerbatim("MuonAssociatorByHits") 
	    << " NOT associated to reco::Track " << tindex  
	    <<" with quality = " << global_quality <<" (tracker: "<<tracker_quality<<" / muon: "<<muon_quality<<")";
	}
	edm::LogVerbatim("MuonAssociatorByHits")
	  <<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";  
      }
    }
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
    LogTrace("MuonAssociatorByHits") << "Unknown Tracker subdetector: subdetId = " <<  subdetId;
  
  return layerNumber;
} 

template<typename iter>
void MuonAssociatorByHits::getMatchedIds(std::vector<SimHitIdpr>& tracker_matchedIds,std::vector<SimHitIdpr>& muon_matchedIds,
					 int& n_valid_hits, int& n_tracker_valid_hits, 
					 int& n_dt_valid_hits, int& n_csc_valid_hits, int& n_rpc_valid_hits,
					 int& n_matched_hits, int& n_tracker_matched_hits, 
					 int& n_dt_matched_hits, int& n_csc_matched_hits, int& n_rpc_matched_hits,
					 iter begin,
					 iter end,
					 TrackerHitAssociator* trackertruth, 
					 DTHitAssociator & dttruth,
					 MuonTruth & csctruth,
					 RPCHitAssociator & rpctruth) const {
    tracker_matchedIds.clear();
    muon_matchedIds.clear();

    n_valid_hits = 0;            // number of valid rechits (Total)
    n_tracker_valid_hits = 0;    //                         (Tracker)
    n_dt_valid_hits = 0;         //                         (DT)
    n_csc_valid_hits = 0;        //                         (CSC)
    n_rpc_valid_hits = 0;        //                         (RPC)

    n_matched_hits = 0;          // number of associated rechits (Total)
    n_tracker_matched_hits = 0;  //                              (Tracker)
    n_dt_matched_hits = 0;       //                              (DT)
    n_csc_matched_hits = 0;      //                              (CSC)
    n_rpc_matched_hits = 0;      //                              (RPC)

    std::vector<SimHitIdpr> SimTrackIds;

    int iloop = 0;
    for (iter it = begin;  it != end; it++, iloop++) {
      unsigned int detid = getHitPtr(it)->geographicalId().rawId();
      DetId::Detector det = getHitPtr(it)->geographicalId().det();
      int subdet = getHitPtr(it)->geographicalId().subdetId();
      
      if (getHitPtr(it)->isValid()) {
	n_valid_hits++;
	
        if (det == DetId::Tracker) {
	  edm::LogVerbatim("MuonAssociatorByHits") 
	    <<"TrackingRecHit "<<iloop<<" -Tracker - detID = "<<detid<<" matched to SimTrack";

	  n_tracker_valid_hits++;
	  SimTrackIds.clear();	  
	  SimTrackIds = trackertruth->associateHitId(*getHitPtr(it));

	  if(!SimTrackIds.empty()) {
	    n_tracker_matched_hits++;
	    n_matched_hits++;
	    for(size_t j=0; j<SimTrackIds.size(); j++){
	      edm::LogVerbatim("MuonAssociatorByHits")
		<<" Id:" << SimTrackIds[j].first 
		<< "/Evt:(" << SimTrackIds[j].second.event()<<","
		<< SimTrackIds[j].second.bunchCrossing()<<") ";
	      tracker_matchedIds.push_back(SimTrackIds[j]);			
	    }
	  }
	}
	else if (det == DetId::Muon) {
	  if (subdet == MuonSubdetId::DT) {
	    edm::LogVerbatim("MuonAssociatorByHits") 
	      <<"TrackingRecHit "<<iloop<<" -Muon DT - detID = "<<detid<<" matched to SimTrack";
	    
	    n_dt_valid_hits++;
	    SimTrackIds.clear();
	    SimTrackIds = dttruth.associateHitId(*getHitPtr(it));  
	    
	    if (!SimTrackIds.empty()) {
	      n_dt_matched_hits++;
	      n_matched_hits++;
	      for(unsigned int j=0; j<SimTrackIds.size(); j++) {
		edm::LogVerbatim("MuonAssociatorByHits")
		  <<" Id:" << SimTrackIds[j].first 
		  << "/Evt:(" << SimTrackIds[j].second.event()<<","
		  << SimTrackIds[j].second.bunchCrossing()<<") ";
		muon_matchedIds.push_back(SimTrackIds[j]);
	      }
	    }
	    if (debug) {
	      const DTRecHit1D * dtrechit = dynamic_cast<const DTRecHit1D *>(&(*getHitPtr(it)));
	      DTWireId wireid = dtrechit->wireId();
	      edm::LogVerbatim("MuonAssociatorByHits") << " DTWireId :" <<wireid;

	      std::vector<const PSimHit *> dtSimHits = dttruth.associateHit(*getHitPtr(it));
	      edm::LogVerbatim("MuonAssociatorByHits") << " associated PSimHit's : size = "<<dtSimHits.size();
	      for (unsigned int j=0; j<dtSimHits.size(); j++) {
		edm::LogVerbatim("MuonAssociatorByHits") << "index = " <<j<< ", PSimHit = "<< *(dtSimHits[j]);
	      }
	    }
	  }
	  
	  else if (subdet == MuonSubdetId::CSC) {
	    edm::LogVerbatim("MuonAssociatorByHits") 
	      <<"TrackingRecHit "<<iloop<<" -Muon CSC- detID = "<<detid<<" matched to SimTrack";

	    n_csc_valid_hits++;
	    SimTrackIds.clear();
	    SimTrackIds = csctruth.associateHitId(*getHitPtr(it));  
	    
	    if (!SimTrackIds.empty()) {
	      n_csc_matched_hits++;
	      n_matched_hits++;
	      for(unsigned int j=0; j<SimTrackIds.size(); j++) {
		edm::LogVerbatim("MuonAssociatorByHits")
		  <<" Id:" << SimTrackIds[j].first 
		  << "/Evt:(" << SimTrackIds[j].second.event()<<","
		  << SimTrackIds[j].second.bunchCrossing()<<") ";
		muon_matchedIds.push_back(SimTrackIds[j]);
	      }
	    } 
	  }
	  
	  else if (subdet == MuonSubdetId::RPC) {
	    edm::LogVerbatim("MuonAssociatorByHits") 
	      <<"TrackingRecHit "<<iloop<<" -Muon RPC- detID = "<<detid<<" matched to SimTrack";

	    n_rpc_valid_hits++;
	    SimTrackIds.clear();
	    SimTrackIds = rpctruth.associateRecHit(*getHitPtr(it));  
	    
	    if (!SimTrackIds.empty()) {
	      n_rpc_matched_hits++;
	      n_matched_hits++;
	      for(unsigned int j=0; j<SimTrackIds.size(); j++) {
		edm::LogVerbatim("MuonAssociatorByHits")
		  <<" Id:" << SimTrackIds[j].first 
		  << "/Evt:(" << SimTrackIds[j].second.event()<<","
		  << SimTrackIds[j].second.bunchCrossing()<<") ";
		muon_matchedIds.push_back(SimTrackIds[j]);
	      }
	    } 
	  }
 	  
	} else edm::LogVerbatim("MuonAssociatorByHits")
	  <<"TrackingRecHit "<<iloop<<"  *** WARNING *** Unexpected Hit from Detector = "<<det;
	
      } else edm::LogVerbatim("MuonAssociatorByHits")
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

