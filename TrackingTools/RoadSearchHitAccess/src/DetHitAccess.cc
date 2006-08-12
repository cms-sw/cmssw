#include "TrackingTools/RoadSearchHitAccess/interface/DetHitAccess.h"

DetHitAccess::DetHitAccess(const SiStripRecHit2DCollection* rphiRecHits,
			   const SiStripRecHit2DCollection* stereoRecHits,
			   const SiStripMatchedRecHit2DCollection* matchedRecHits,
			   const SiPixelRecHitCollection* pixelRecHits) {
  
  rphiHits_ = rphiRecHits;
  stereoHits_ = stereoRecHits;
  matchedHits_ = matchedRecHits;
  pixelHits_ = pixelRecHits;
  
}
		

edm::OwnVector<TrackingRecHit> DetHitAccess::getHitVector(const DetId* detid) {
  
  edm::OwnVector<TrackingRecHit> TrkRecHitVec;
  
  if ( (unsigned int)detid->subdetId() == StripSubdetector::TIB  
       || (unsigned int)detid->subdetId() == StripSubdetector::TOB  
       || (unsigned int)detid->subdetId() == StripSubdetector::TID  
       || (unsigned int)detid->subdetId() == StripSubdetector::TEC )  { 
    
    uint32_t detId_tmp = detid->rawId();
    StripSubdetector StripDetId(*detid);

    uint32_t tec_ring=0;
    if((uint32_t)detid->subdetId() == StripSubdetector::TEC) {
      TECDetId tec_detid(*detid);
      tec_ring = tec_detid.ring();
    }
    
    bool use_matched_id = true;
    if ( (uint32_t)detid->subdetId() == StripSubdetector::TOB)                use_matched_id = false;
    if ( (uint32_t)detid->subdetId() == StripSubdetector::TEC && tec_ring>2)  use_matched_id = false;


    if ( StripDetId.glued() && use_matched_id ) {

      //DetId is based on rphi -> used DetId-1 to get matchedHits
      //DetId usedDetId(detId_tmp);
      DetId usedDetId(StripDetId.glued());
      
      SiStripMatchedRecHit2DCollection::range matchedDetHits = matchedHits_->get(usedDetId);
      //if((unsigned int)detid->subdetId() == StripSubdetector::TIB) edm::LogError("RoadSearch") << "TIB glued: " << usedDetId.rawId() << " GLUED ";
      //if((unsigned int)detid->subdetId() == StripSubdetector::TOB) edm::LogError("RoadSearch") << "TOB glued: " << usedDetId.rawId() << " GLUED ";
      //if((unsigned int)detid->subdetId() == StripSubdetector::TID) edm::LogError("RoadSearch") << "TID glued: " << usedDetId.rawId() << " GLUED ";
      //if((unsigned int)detid->subdetId() == StripSubdetector::TEC) edm::LogError("RoadSearch") << "TEC glued: " << usedDetId.rawId() << " GLUED ";

      // loop over inner dethits
      for ( SiStripMatchedRecHit2DCollection::const_iterator matchedDetHit = matchedDetHits.first;
	    matchedDetHit != matchedDetHits.second; ++matchedDetHit ) {
	//edm::LogError("RoadSearch") << "matched: " << (matchedDetHit->geographicalId()).rawId();
	TrkRecHitVec.push_back((TrackingRecHit*)(matchedDetHit->clone()));

      }
      
    } else {
      DetId usedDetId(detId_tmp);

      //if((unsigned int)detid->subdetId() == StripSubdetector::TIB) edm::LogError("RoadSearch") << "TIB: " << usedDetId.rawId() << " NOT GLUED ";
      //if((unsigned int)detid->subdetId() == StripSubdetector::TOB) edm::LogError("RoadSearch") << "TOB: " << usedDetId.rawId() << " NOT GLUED ";
      //if((unsigned int)detid->subdetId() == StripSubdetector::TID) edm::LogError("RoadSearch") << "TID: " << usedDetId.rawId() << " NOT GLUED ";
      //if((unsigned int)detid->subdetId() == StripSubdetector::TEC) edm::LogError("RoadSearch") << "TEC: " << usedDetId.rawId() << " NOT GLUED ";


      SiStripRecHit2DCollection::range rphiDetHits = rphiHits_->get(usedDetId);
      for ( SiStripRecHit2DCollection::const_iterator rphiDetHit = rphiDetHits.first;
	    rphiDetHit != rphiDetHits.second; ++rphiDetHit ) {
	//edm::LogError("RoadSearch") << "CARSTEN: " << (rphiDetHit->geographicalId()).rawId();
	TrkRecHitVec.push_back((TrackingRecHit*)(rphiDetHit->clone()));
      }
	
    }
    
  } else if ( (unsigned int)detid->subdetId() == PixelSubdetector::PixelBarrel 
	      || (unsigned int)detid->subdetId() == PixelSubdetector::PixelEndcap) {
    
    //edm::LogError("RoadSearch") << "PIXEL ALGORITHM";
    SiPixelRecHitCollection::range pixelDetHits = pixelHits_->get(*detid);
    for ( SiPixelRecHitCollection::const_iterator pixelDetHit = pixelDetHits.first; 
	  pixelDetHit!= pixelDetHits.second; ++pixelDetHit) {
      
      TrkRecHitVec.push_back((TrackingRecHit*)(pixelDetHit->clone()));
    }
    
  } else {

    //edm::LogError("RoadSearch") << "NEITHER PIXEL NOR STRIP ALGO RUN";

  }
  //edm::LogError("RoadSearch") << "TrkRecHitVec.size(): " << TrkRecHitVec.size();
  return TrkRecHitVec;
  //Q for Oliver:
  //StripSubdetector::StripSubdetector(const DetId&)
  //StripSubdetector::StripSubdetector(detid) a;
  
}
  
