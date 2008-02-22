//
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//reco track
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
//TrackingParticle
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
//##---new stuff
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
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

/* Constructor */
TrackAssociatorByHits::TrackAssociatorByHits (const edm::ParameterSet& conf) :  
  conf_(conf),
  AbsoluteNumberOfHits(conf_.getParameter<bool>("AbsoluteNumberOfHits")),
  theMinHitCut(conf_.getParameter<double>("MinHitCut"))
{
}


/* Destructor */
TrackAssociatorByHits::~TrackAssociatorByHits()
{
  //do cleanup here
}

//
//---member functions
//

RecoToSimCollection  
TrackAssociatorByHits::associateRecoToSim(edm::Handle<reco::TrackCollection>& trackCollectionH,
					  edm::Handle<TrackingParticleCollection>&  TPCollectionH,     
					  const edm::Event * e ) const{

  edm::LogVerbatim("TrackAssociator") << "Starting TrackAssociatorByHits::associateRecoToSim";
  int nshared = 0;
  float quality=0;//fraction or absolute number of shared hits
  std::vector< SimHitIdpr> SimTrackIds;
  std::vector< SimHitIdpr> matchedIds; 
  RecoToSimCollection  outputCollection;
  
  TrackerHitAssociator * associate = new TrackerHitAssociator::TrackerHitAssociator(*e, conf_);
  
  const TrackingParticleCollection tPC   = *(TPCollectionH.product());

  const  reco::TrackCollection  tC = *(trackCollectionH.product()); 
  
  
  //get the ID of the recotrack  by hits 
  int tindex=0;
  for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++, tindex++) {
    matchedIds.clear();
    int ri=0;//valid rechits
    for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
      if ((*it)->isValid()){
	ri++;
	uint32_t t_detID=  (*it)->geographicalId().rawId();
	SimTrackIds.clear();	  
	SimTrackIds = associate->associateHitId((**it));
	//save all the id of matched simtracks
	//*** simple version
	if(!SimTrackIds.empty()){
	 for(size_t j=0; j<SimTrackIds.size(); j++){
	   LogTrace("TrackAssociator") << " hit # " << ri << " valid=" << (*it)->isValid() 
				       << " det id = " << t_detID << " SimId " << SimTrackIds[j].first 
				       << " evt=" << SimTrackIds[j].second.event() 
				       << " bc=" << SimTrackIds[j].second.bunchCrossing();  
	   matchedIds.push_back(SimTrackIds[j]);			
	 }
	}

	//*** this could help to avoid double counting of hits with the same EncodedEvent
	// 	if(SimTrackIds.size()==1){
	//  LogTrace("TrackAssociator") << " hit # " << ri << " SimId " << SimTrackIds[0].first 
	//			      << " event id = " << SimTrackIds[0].second.event() 
	//			      << " Bunch Xing = " << SimTrackIds[0].second.bunchCrossing(); 
	//  matchedIds.push_back(SimTrackIds[0]);
	//}
	//else if (SimTrackIds.size()>1) {
	//  LogTrace("TrackAssociator") << "SimTrackIds.size()>1";
	//  std::vector<SimHitIdpr> tmpIds;
	//  for(size_t j=0; j<SimTrackIds.size(); j++) {
	//    LogTrace("TrackAssociator") << " hit # " << ri << " SimId " << SimTrackIds[j].first 
	//				<< " event id = " << SimTrackIds[j].second.event() 
	//				<< " Bunch Xing = " << SimTrackIds[j].second.bunchCrossing(); 
	//    bool newevtid = true;
	//    for(size_t jj=0; jj<tmpIds.size(); jj++){//avoid double counting of hits with the same EncodedEvent
	//      if (SimTrackIds[j].second.event()==tmpIds[jj].second.event()
	//	  && SimTrackIds[j].second.bunchCrossing()==tmpIds[jj].second.bunchCrossing()){
	//	newevtid = false;
	//      }
	//    }
	//    if (newevtid) {
	//      LogTrace("TrackAssociator") << "pushing " << SimTrackIds[j].first;
	//      tmpIds.push_back(SimTrackIds[j]);
	//    }
	//    if (find(matchedIds.begin(), matchedIds.end(),SimTrackIds[j])!=matchedIds.end()){
	//      LogTrace("TrackAssociator") << "pushing " << SimTrackIds[j].first;
	//      tmpIds.push_back(SimTrackIds[j]);
	//      for (std::vector<SimHitIdpr>::iterator jj=tmpIds.begin(); jj<tmpIds.end()-1; jj++) {
	//	if (SimTrackIds[j].second.event()==jj->second.event()
	//	    && SimTrackIds[j].second.bunchCrossing()==jj->second.bunchCrossing()) {
	//	  LogTrace("TrackAssociator") << "erasing " << jj->first;
	//	  tmpIds.erase(jj);
	//	}
	//      }
	//    }
	//  }
	//  matchedIds.insert(matchedIds.end(),tmpIds.begin(),tmpIds.end());
	//}
	
      }else{
	LogTrace("TrackAssociator") <<"\t\t Invalid Hit On "<<(*it)->geographicalId().rawId();
      }
    }//trackingRecHit loop

    //LogTrace("TrackAssociator") << "MATCHED IDS LIST BEGIN" ;
    //for(size_t j=0; j<matchedIds.size(); j++){
    //  LogTrace("TrackAssociator") << "matchedIds[j].first=" << matchedIds[j].first;
    //}
    //LogTrace("TrackAssociator") << "MATCHED IDS LIST END" ;

    LogTrace("TrackAssociator") << "#matched ids=" << matchedIds.size() << " #tps=" << tPC.size();
    //save id for the track
    std::vector<SimHitIdpr> idcachev;
    if(!matchedIds.empty()){

      int tpindex =0;
      for (TrackingParticleCollection::const_iterator t = tPC.begin(); t != tPC.end(); ++t, ++tpindex) {
	LogTrace("TrackAssociator") << "TP #" << tpindex;
	idcachev.clear();
	nshared =0;
	for(size_t j=0; j<matchedIds.size(); j++){
	  LogTrace("TrackAssociator") << "now matchedId=" << matchedIds[j].first;
	  //replace with a find in vector
	  if(find(idcachev.begin(), idcachev.end(),matchedIds[j]) == idcachev.end() ){
	    //only the first time we see this ID 
	    idcachev.push_back(matchedIds[j]);

	    for (TrackingParticle::g4t_iterator g4T = t -> g4Track_begin(); g4T !=  t -> g4Track_end(); ++g4T) {
	      LogTrace("TrackAssociator") << " TP   (ID, Ev, BC) = " << (*g4T).trackId() 
					  << ", " << t->eventId().event() << ", "<< t->eventId().bunchCrossing()
					  << " Match(ID, Ev, BC) = " <<  matchedIds[j].first
					  << ", " << matchedIds[j].second.event() << ", "
					  << matchedIds[j].second.bunchCrossing() 
					  << "\t G4  Track Momentum " << (*g4T).momentum() 
					  << " \t reco Track Momentum " << track->momentum();  	      
	      if((*g4T).trackId() == matchedIds[j].first && t->eventId() == matchedIds[j].second){
		nshared += std::count(matchedIds.begin(), matchedIds.end(), matchedIds[j]);
	      }
	    }//g4Tracks loop
	  }
	}
	
	if (AbsoluteNumberOfHits) quality = static_cast<double>(nshared);
	else if(ri!=0) quality = (static_cast<double>(nshared)/static_cast<double>(ri));
	else quality = 0;
	//for now save the number of shared hits between the reco and sim track
	//cut on the fraction
	if(quality > theMinHitCut){
	  if(!AbsoluteNumberOfHits && quality>1.) std::cout << " **** fraction > 1 " << " nshared = " << nshared 
							    << "rechits = " << ri << " hit found " << track->found() <<  std::endl;
	  outputCollection.insert(reco::TrackRef(trackCollectionH,tindex), 
				  std::make_pair(edm::Ref<TrackingParticleCollection>(TPCollectionH, tpindex),
						 quality));
	  LogTrace("TrackAssociator") << "reco::Track number " << tindex  
				      << "associated to TP (pdgId, nb segments, p) = " 
				      << (*t).pdgId() << " " << (*t).g4Tracks().size() 
				      << " " << (*t).momentum() << " with quality =" << quality;
	} else {
	  LogTrace("TrackAssociator") <<"reco::Track number " << tindex << " NOT associated with quality =" << quality;
	}
      }//TP loop
    }
  }
  LogTrace("TrackAssociator") << "% of Assoc Tracks=" << ((double)outputCollection.size())/((double)trackCollectionH->size());
  delete associate;
  outputCollection.post_insert();
  return outputCollection;
}


SimToRecoCollection  
TrackAssociatorByHits::associateSimToReco(edm::Handle<reco::TrackCollection>& trackCollectionH,
					  edm::Handle<TrackingParticleCollection>&  
					  TPCollectionH, 
					  const edm::Event * e ) const{
  edm::LogVerbatim("TrackAssociator") << "Starting TrackAssociatorByHits::associateSimToReco";
  float quality=0;//fraction or absolute number of shared hits
  int nshared = 0;
  std::vector< SimHitIdpr> SimTrackIds;
  std::vector< SimHitIdpr> matchedIds; 
  SimToRecoCollection  outputCollection;

  TrackerHitAssociator * associate = new TrackerHitAssociator::TrackerHitAssociator(*e, conf_);
  
  const TrackingParticleCollection tPC   = *(TPCollectionH.product());

  //for (TrackingParticleCollection::const_iterator t = tPC.begin(); t != tPC.end(); ++t) {
  //  LogTrace("TrackAssociator") << "NEW TP DUMP";
  //  for (TrackingParticle::g4t_iterator g4T = t -> g4Track_begin();g4T !=  t -> g4Track_end(); ++g4T) {
  //    LogTrace("TrackAssociator") << "(*g4T).trackId()=" <<(*g4T).trackId() ;
  //  }
  //}
  
  const  reco::TrackCollection  tC = *(trackCollectionH.product()); 

  //get the ID of the recotrack  by hits 
  int tindex=0;
  for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++, tindex++) {
    LogTrace("TrackAssociator") << " hits of track with pt =" << track->pt() << " # valid=" << track->found(); 
    matchedIds.clear();
    int ri=0;
    for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
      if ((*it)->isValid()){
	ri++;
	DetId t_detid=  (*it)->geographicalId();
	uint32_t t_detID = t_detid.rawId();
	SimTrackIds.clear();	  
	SimTrackIds = associate->associateHitId((**it));

	//save all the id of matched simtracks
	//*** simple version
	if(!SimTrackIds.empty()){
	 for(size_t j=0; j<SimTrackIds.size(); j++){
	   LogTrace("TrackAssociator") << " hit # " << ri << " valid=" << (*it)->isValid() 
				       << " det id = " << t_detID << " SimId " << SimTrackIds[j].first 
				       << " evt=" << SimTrackIds[j].second.event() 
				       << " bc=" << SimTrackIds[j].second.bunchCrossing();  
	   matchedIds.push_back(SimTrackIds[j]);			
	 }
	}

	//*** this could help to avoid double counting of hits with the same EncodedEvent
	//if(SimTrackIds.size()==1){
	//  LogTrace("TrackAssociator") << " hit # " << ri << " valid=" << (*it)->isValid() 
	//			      << " det id = " << t_detID << " SUBDET = " << t_detid.subdetId() 
	//			      << " layer = " << LayerFromDetid(t_detID) 
	//			      << " SimId " << SimTrackIds[0].first 
	//			      << " evt=" << SimTrackIds[0].second.event() 
	//			      << " bc=" << SimTrackIds[0].second.bunchCrossing();  
	//  matchedIds.push_back(SimTrackIds[0]);
	//}
	//else if (SimTrackIds.size()>1) {
	//  LogTrace("TrackAssociator") << "SimTrackIds.size()>1";
	//  std::vector<SimHitIdpr> tmpIds;
	//  for(size_t j=0; j<SimTrackIds.size(); j++) {
	//    LogTrace("TrackAssociator") << " hit # " << ri << " valid=" << (*it)->isValid() 
	//				<< " det id = " << t_detID << " SUBDET = " << t_detid.subdetId() 
	//				<< " layer = " << LayerFromDetid(t_detID) 
	//				<< " SimId " << SimTrackIds[j].first 
	//				<< " evt=" << SimTrackIds[j].second.event() 
	//				<< " bc=" << SimTrackIds[j].second.bunchCrossing();  
	//    bool newevtid = true;
	//    for(size_t jj=0; jj<tmpIds.size(); jj++){//avoid double counting of hits with the same EncodedEvent
	//      if (SimTrackIds[j].second.event()==tmpIds[jj].second.event()
	//	  && SimTrackIds[j].second.bunchCrossing()==tmpIds[jj].second.bunchCrossing()){
	//	newevtid = false;
	//      }
	//    }
	//    if (newevtid) {
	//      LogTrace("TrackAssociator") << "pushing " << SimTrackIds[j].first;
	//      tmpIds.push_back(SimTrackIds[j]);
	//    }
	//    if (find(matchedIds.begin(), matchedIds.end(),SimTrackIds[j])!=matchedIds.end()){
	//      LogTrace("TrackAssociator") << "pushing " << SimTrackIds[j].first;
	//      tmpIds.push_back(SimTrackIds[j]);
	//      for (std::vector<SimHitIdpr>::iterator jj=tmpIds.begin(); jj<tmpIds.end()-1; jj++) {
	//	if (SimTrackIds[j].second.event()==jj->second.event()
	//	    && SimTrackIds[j].second.bunchCrossing()==jj->second.bunchCrossing()) {
	//	  LogTrace("TrackAssociator") << "erasing " << jj->first;
	//	  tmpIds.erase(jj);
	//	}
	//      }
	//    }
	//  }
	//  matchedIds.insert(matchedIds.end(),tmpIds.begin(),tmpIds.end());
	//}
	
      }else{
	LogTrace("TrackAssociator") <<"\t\t Invalid Hit On "<<(*it)->geographicalId().rawId();
      }
    }

    //save id for the track
    std::vector<SimHitIdpr> idcachev;
    if(!matchedIds.empty()){
	
      int tpindex =0;
      for (TrackingParticleCollection::const_iterator t = tPC.begin(); t != tPC.end(); ++t, ++tpindex) {
	LogTrace("TrackAssociator") << "NEW TP";
	LogTrace("TrackAssociator") << "number of PSimHits for this TP: "  << t->trackPSimHit().size() ;
	idcachev.clear();
	nshared =0;
	int nsimhit = 0;
	float totsimhit = 0; 
	std::vector<PSimHit> tphits;
	for(size_t j=0; j<matchedIds.size(); j++){
	  //replace with a find in vector
	  LogTrace("TrackAssociator") << "now matchedId=" << matchedIds[j].first;
	  if(find(idcachev.begin(), idcachev.end(),matchedIds[j]) == idcachev.end() ){
	    //only the first time we see this ID 
	    idcachev.push_back(matchedIds[j]);
	    for (TrackingParticle::g4t_iterator g4T = t -> g4Track_begin(); g4T !=  t -> g4Track_end(); ++g4T) {
	      if((*g4T).trackId() == matchedIds[j].first && t->eventId() == matchedIds[j].second) {
		//LogTrace("TrackAssociator") << "hits of this segment";
		//for(std::vector<PSimHit>::const_iterator TPhit = t->pSimHit_begin(); TPhit != t->pSimHit_end(); TPhit++){
		//  DetId dId = DetId(TPhit->detUnitId());
		//  LogTrace("TrackAssociator") << "hit SUBDET = " << dId.subdetId() 
		//				      << " layer = " << LayerFromDetid(dId) << " id=" << dId.rawId();
		//}
		LogTrace("TrackAssociator") << " TP   (pdgId, ID, Ev, BC) = " 
					    << (*g4T).type() << " " << (*g4T).trackId() 
					    << ", " << t->eventId().event() << ", "<< t->eventId().bunchCrossing(); 
		LogTrace("TrackAssociator") << " Match(ID, Ev, BC) = " <<  matchedIds[j].first
					    << ", " << matchedIds[j].second.event() << ", "
					    << matchedIds[j].second.bunchCrossing() 
					    << "\n G4  Track Momentum " << (*g4T).momentum() 
					    << "\t reco Track Pt " << track->pt();  

		int countedhits = std::count(matchedIds.begin(), matchedIds.end(), matchedIds[j]);
		nshared += countedhits;

		LogTrace("TrackAssociator") << "hits shared by this segment : " << countedhits;
		LogTrace("TrackAssociator") << "hits shared so far : " << nshared;
		  
		nsimhit += t->trackPSimHit().size(); 
	      }
	    }//g4Tracks loop
	  }
	}//matchedIds loop

	//for(std::vector<PSimHit>::const_iterator TPhit = t->pSimHit_begin(); TPhit != t->pSimHit_end(); TPhit++){
	//  unsigned int detid = TPhit->detUnitId();
	//  DetId detId = DetId(TPhit->detUnitId());
	//  LogTrace("TrackAssociator") <<  " hit trackId= " << TPhit->trackId() << " det ID = " << detid 
	//				      << " SUBDET = " << detId.subdetId() << " layer = " << LayerFromDetid(detId); 
	//}

	//count the TP simhit, counting only once the hits on glued detectors
	//should change for grouped trajectory builder
	LogTrace("TrackAssociator") << "recounting of tp hits";
	for(std::vector<PSimHit>::const_iterator TPhit = t->pSimHit_begin(); TPhit != t->pSimHit_end(); TPhit++){
	  DetId dId = DetId(TPhit->detUnitId());
	  LogTrace("TrackAssociator") << "consider hit SUBDET = " << dId.subdetId() 
				      << " layer = " << LayerFromDetid(dId) << " id=" << dId.rawId();
	  bool newhit = true;		  
	  for(std::vector<PSimHit>::const_iterator TPhitOK = tphits.begin(); TPhitOK != tphits.end(); TPhitOK++){
	    DetId dIdOK = DetId(TPhitOK->detUnitId());
	    LogTrace("TrackAssociator") << "\t\tcompare with SUBDET = " << dIdOK.subdetId() 
					<< " layer = " << LayerFromDetid(dIdOK);
	    if (LayerFromDetid(dId)==LayerFromDetid(dIdOK) && 
		dId.subdetId()==dIdOK.subdetId()) newhit = false;
	  }
	  if (newhit) {
	    LogTrace("TrackAssociator") << "ok";
	    tphits.push_back(*TPhit);
	  }
	  else LogTrace("TrackAssociator") << "no";
	}

	////count the TP simhit, counting only once the hits on glued detectors
	//float totsimhitlay = 0; 
	////int glue_cache = 0;	
	////counting the TP hits using the layers (as in ORCA). 
	////does seem to find less hits. maybe b/c layer is a number now, not a pointer
	//int oldlay = 0;
	//int newlay = 0;
	//int olddet = 0;
	//int newdet = 0;
	//for(std::vector<PSimHit>::const_iterator TPhit = t->pSimHit_begin(); TPhit != t->pSimHit_end(); TPhit++){
	//  unsigned int detid = TPhit->detUnitId();
	//  DetId detId = DetId(TPhit->detUnitId());
	//  oldlay = newlay;
	//  olddet = newdet;
	//  newlay = LayerFromDetid(detId);
	//  newdet = detId.subdetId();
	//  if(oldlay != newlay || (oldlay==newlay && olddet!=newdet) ){
	//    totsimhitlay++;
	//    LogTrace("TrackAssociator") <<  " hit = " << TPhit->trackId() << " det ID = " << detid 
	//    			  << " SUBDET = " << detId.subdetId() << "layer = " << LayerFromDetid(detId); 
	//  }  
	//}//loop over TP simhit       
	//totsimhitP = totsimhitlay;
	
	totsimhit = tphits.size();

	if (AbsoluteNumberOfHits) quality = static_cast<double>(nshared);
	else if(totsimhit!=0) quality = ((double) nshared)/((double)totsimhit);
	else quality = 0;
	LogTrace("TrackAssociator") << "Final count: nhit(TP) = " << nsimhit << " re-counted = " << totsimhit 
				    << " re-count(lay) = " << totsimhit << " nshared = " << nshared 
				    << " nrechit = " << ri;
	if (quality>theMinHitCut) {
	  outputCollection.insert(edm::Ref<TrackingParticleCollection>(TPCollectionH, tpindex), 
				  std::make_pair(reco::TrackRef(trackCollectionH,tindex),quality));
	  edm::LogVerbatim("TrackAssociator") << "TrackingParticle number " << tpindex 
					      << " associated to track with pt=" << track->pt() 
					      << " with hit quality =" << quality ;
	}
	else {
	  LogTrace("TrackAssociator") << "TrackingParticle number " << tpindex << " NOT associated with quality =" << quality;
	}
      }
    }
  }
  LogTrace("TrackAssociator") << "% of Assoc TPs=" << ((double)outputCollection.size())/((double)TPCollectionH->size());
  delete associate;
  outputCollection.post_insert();
  return outputCollection;
}

int TrackAssociatorByHits::LayerFromDetid(const DetId& detId ) const
{
  int layerNumber=0;
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
    LogTrace("TrackAssociator") << "Unknown subdetid: " <<  subdetId;
  
  return layerNumber;
} 

