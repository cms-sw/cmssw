//
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
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
  theMinHitFraction(conf_.getParameter<double>("MinHitFraction"))
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
					  const edm::Event * e ){

  const double minHitFraction = theMinHitFraction;
  int nshared =0;
  float fraction=0;
  std::vector<unsigned int> SimTrackIds;
  std::vector<unsigned int> matchedIds; 
  RecoToSimCollection  outputCollection;
  
  TrackerHitAssociator * associate = new TrackerHitAssociator::TrackerHitAssociator(*e, conf_);

  const TrackingParticleCollection tPC   = *(TPCollectionH.product());

  const  reco::TrackCollection  tC = *(trackCollectionH.product()); 


  //get the ID of the recotrack  by hits 
  int tindex=0;
  for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++, tindex++)
    {
      matchedIds.clear();
      int ri=0;//valid rechits
      for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
	if ((*it)->isValid()){
	  ri++;
	  SimTrackIds.clear();	  
	  SimTrackIds = associate->associateHitId((**it));
	  //save all the id of matched simtracks
	  if(!SimTrackIds.empty()){
	    for(size_t j=0; j<SimTrackIds.size(); j++){
	      // std::cout << " hit # " << ri << " SimId " << SimTrackIds[j] << std::endl; 
	      matchedIds.push_back(SimTrackIds[j]);
	    }
	  }
	}else{
	  edm::LogVerbatim("TrackValidator") <<"\t\t Invalid Hit On "<<(*it)->geographicalId().rawId();
	}
      }
      //save id for the track
      std::vector<unsigned int> idcachev;
      if(!matchedIds.empty()){
	idcachev.push_back(9999999);
	nshared =0;
	fraction =0;
	for(size_t j=0; j<matchedIds.size(); j++){
	  //replace with a find in vector
	  if(find(idcachev.begin(), idcachev.end(),matchedIds[j]) == idcachev.end() ){
	    //only the first time we see this ID 
	    idcachev.push_back(matchedIds[j]);
	    int tpindex =0;
	    for (TrackingParticleCollection::const_iterator t = tPC.begin(); t != tPC.end(); ++t, ++tpindex) 
	      {
		for (TrackingParticle::g4t_iterator g4T = t -> g4Track_begin();
		     g4T !=  t -> g4Track_end(); ++g4T) {
		  if((*g4T).trackId() == matchedIds[j]){
		    // 		    std::cout << " Match ID = " << (*g4T)->trackId() 
		    // 			      << " Nrh = " << ri << " Nshared = " << n << std::endl;
		    // 		    std::cout << " G4  Track Momentum " << (*g4T)->momentum() << std::endl;   
		    // 		    std::cout << " reco Track Momentum " << track->momentum() << std::endl;  
		    nshared = std::count(matchedIds.begin(), matchedIds.end(), matchedIds[j]);
		    if(ri!=0) fraction = (static_cast<double>(nshared)/static_cast<double>(ri));
		    //for now save the number of shared hits between the reco and sim track
		    //cut on the fraction
		    if(fraction>minHitFraction){
		      if(fraction>1.) std::cout << " **** fraction >1 " << " nshared = " << nshared 
						<< "rechits = " << ri << " hit found " << track->found() <<  std::endl;
		      outputCollection.insert(reco::TrackRef(trackCollectionH,tindex), 
					      std::make_pair(edm::Ref<TrackingParticleCollection>(TPCollectionH, tpindex),
							     fraction));
		      edm::LogVerbatim("TrackValidator") <<"reco::Track number " << tindex  << " associated with hit fraction =" << fraction;
		    } else {
		      edm::LogVerbatim("TrackValidator") <<"reco::Track number " << tindex << " NOT associated with hit fraction =" << fraction;
		    }
		  }
		}
	      }
	  }
	}
      }
    }
  delete associate;
  return outputCollection;
}


SimToRecoCollection  
TrackAssociatorByHits::associateSimToReco(edm::Handle<reco::TrackCollection>& trackCollectionH,
					  edm::Handle<TrackingParticleCollection>&  
					  TPCollectionH, 
					  const edm::Event * e ){
  
  const double minHitFraction = theMinHitFraction;
  float fraction=0;
  int nshared = 0;
  std::vector<unsigned int> SimTrackIds;
  std::vector<unsigned int> matchedIds; 
  SimToRecoCollection  outputCollection;

  TrackerHitAssociator * associate = new TrackerHitAssociator::TrackerHitAssociator(*e, conf_);
  
  const TrackingParticleCollection tPC   = *(TPCollectionH.product());
//   std::cout << "Found " << tPC.size() << " TrackingParticles" << std::endl;
  
  const  reco::TrackCollection  tC = *(trackCollectionH.product()); 
//   std::cout << "Reconstructed "<< tC.size() << " tracks" << std::endl ;

  //get the ID of the recotrack  by hits 

  int tindex=0;
  for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++, tindex++)
    {
      matchedIds.clear();
      int ri=0;
      for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
	  if ((*it)->isValid()){
	    ri++;
	    //	    DetId t_detid=  (*it)->geographicalId();
	    //uint32_t t_detID = t_detid.rawId();
	    SimTrackIds.clear();	  
	    SimTrackIds = associate->associateHitId((**it));
	    if(!SimTrackIds.empty()){
	      for(size_t j=0; j<SimTrackIds.size(); j++){
		//std::cout << " hit # " << ri << "det id = " << t_detID << " SimId " << SimTrackIds[j] << std::endl; 
		matchedIds.push_back(SimTrackIds[j]);
	      }
	    }
	  }else{
	    edm::LogVerbatim("TrackValidator") <<"\t\t Invalid Hit On "<<(*it)->geographicalId().rawId();
	  }
      }
      //save id for the track
      std::vector<unsigned int> idcachev;
      if(!matchedIds.empty()){
	idcachev.push_back(9999999);
	nshared =0;
	for(size_t j=0; j<matchedIds.size(); j++){
	  //replace with a find in vector
	  if(find(idcachev.begin(), idcachev.end(),matchedIds[j]) == idcachev.end() ){
	    //only the first time we see this ID 
	    idcachev.push_back(matchedIds[j]);
	    int tpindex =0;
	    for (TrackingParticleCollection::const_iterator t = tPC.begin(); t != tPC.end(); ++t, ++tpindex) {
	      for (TrackingParticle::g4t_iterator g4T = t -> g4Track_begin();
		   g4T !=  t -> g4Track_end(); ++g4T) {
		if((*g4T).trackId() == matchedIds[j]){
		  nshared = std::count(matchedIds.begin(), matchedIds.end(), matchedIds[j]);
		  int nsimhit = t->trackPSimHit().size(); 
		  
		  //count the TP simhit, counting only once the hits on glued detectors
		  float totsimhit = 0; 
		  float totsimhitlay = 0; 
		  int glue_cache = 0;

		  //counting the TP hits using the layers (as in ORCA). 
		  //does seem to find less hits. maybe b/c layer is a number now, not a pointer
		  int oldlay = 0;
		  int newlay = 0;
		  int olddet = 0;
		  int newdet = 0;
		  for(std::vector<PSimHit>::const_iterator TPhit = t->pSimHit_begin(); TPhit != t->pSimHit_end(); TPhit++){
		    unsigned int detid = TPhit->detUnitId();
		    DetId detId = DetId(TPhit->detUnitId());
		    oldlay = newlay;
		    olddet = newdet;
		    newlay = LayerFromDetid(detId);
		    newdet = detId.subdetId();
		    if(oldlay !=newlay || (oldlay==newlay && olddet!=newdet) ){
		      totsimhitlay++;
		      edm::LogVerbatim("TrackValidator") <<  " hit = " << TPhit->trackId() << " det ID = " << detid 
							 << " SUBDET = " << detId.subdetId() << "layer = " << LayerFromDetid(detId); 
		    }
		    
// 		    //======counting using glued detector flag =====
// 		    //
// 		    if(detId.subdetId() == StripSubdetector::TIB ||
// 		       detId.subdetId() == StripSubdetector::TOB || 
// 		       detId.subdetId() == StripSubdetector::TID ||
// 		       detId.subdetId() == StripSubdetector::TEC) {
// 		      StripSubdetector stripId = detId;
// 		      unsigned int glued = stripId.glued();
// 		      if(glued!=0){
// 			if(glued != glue_cache) {
// 			  glue_cache = glued;
// 			  totsimhit++;
// 			  //std::cout << " hit = " << TPhit->trackId() << " det ID = " << detid 
// 			  //<< " SUBDET = " << detId.subdetId() << " glued = " << glued << " tothit = " << totsimhit << std::endl;
// 			} else {
// 			  //std::cout << " hit = " << TPhit->trackId() << " det ID = " << detid 
// 			  //<<  " SUBDET = " << detId.subdetId() << " glued = " << glued << " tothit = " << totsimhit << std::endl;    
// 			}
// 		      }else {
// 			totsimhit++;
// 			//std::cout << " hit = " << TPhit->trackId() << " det ID = " << detid 
// 			//<<  " SUBDET = " << detId.subdetId() << " glued = " << glued << " tothit = " << totsimhit << std::endl;
// 		      }
// 		    } 
// 		    else if( detId.subdetId() == PixelSubdetector::PixelBarrel || 
// 			     detId.subdetId() == PixelSubdetector::PixelEndcap) {
// 		      totsimhit++;
// 		      //std::cout << " hit = " << TPhit->trackId() << " det ID = " << detid << " pixel! "  
// 		      //<< " tothit = " << totsimhit << std::endl;
// 		    }
		    
		  }//loop over TP simhit
		  
		  totsimhit = totsimhitlay;
		  if(totsimhit!=0) fraction = ((double) nshared)/((double)totsimhit);
		  edm::LogVerbatim("TrackValidator") << "Final count: nhit(TP) = " << nsimhit << " re-counted = " << totsimhit 
						     << "re-count(lay) = " << totsimhitlay << " nshared = " << nshared << " nrechit = " << ri;
		  if (fraction>minHitFraction) {
		    outputCollection.insert(edm::Ref<TrackingParticleCollection>(TPCollectionH, tpindex), 
					    std::make_pair(reco::TrackRef(trackCollectionH,tindex),fraction));
		    edm::LogVerbatim("TrackValidator") << "TrackingParticle number " << tpindex << " associated with hit fraction =" << fraction;
		  }
		  else {
		    edm::LogVerbatim("TrackValidator") << "TrackingParticle number " << tpindex << " NOT associated with fraction =" << fraction;
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  delete associate;
  return outputCollection;
}

int TrackAssociatorByHits::LayerFromDetid(const DetId& detId )
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
    edm::LogVerbatim("TrackValidator") << "Unknown subdetid: " <<  subdetId;
  
  return layerNumber;
} 
