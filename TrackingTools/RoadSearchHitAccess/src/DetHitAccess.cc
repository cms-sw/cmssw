#include "TrackingTools/RoadSearchHitAccess/interface/DetHitAccess.h"

DetHitAccess::DetHitAccess(){
};

DetHitAccess::~DetHitAccess(){
};

DetHitAccess::DetHitAccess(const SiStripRecHit2DCollection* rphiRecHits,
			   const SiStripRecHit2DCollection* stereoRecHits,
			   const SiStripMatchedRecHit2DCollection* matchedRecHits,
			   const SiPixelRecHitCollection* pixelRecHits) {
  
  rphiHits_ = rphiRecHits;
  stereoHits_ = stereoRecHits;
  matchedHits_ = matchedRecHits;
  pixelHits_ = pixelRecHits;
  
}
		
void DetHitAccess::setCollections(const SiStripRecHit2DCollection* rphiRecHits,
				  const SiStripRecHit2DCollection* stereoRecHits,
				  const SiStripMatchedRecHit2DCollection* matchedRecHits,
				  const SiPixelRecHitCollection* pixelRecHits) {
  
  rphiHits_ = rphiRecHits;
  stereoHits_ = stereoRecHits;
  matchedHits_ = matchedRecHits;
  pixelHits_ = pixelRecHits;
  
}

std::vector<TrackingRecHit*> DetHitAccess::getHitVector(const DetId* detid) {
  
  std::vector<TrackingRecHit*> RecHitVec;
  
  if (    (unsigned int)detid->subdetId() == StripSubdetector::TIB  
       || (unsigned int)detid->subdetId() == StripSubdetector::TOB  
       || (unsigned int)detid->subdetId() == StripSubdetector::TID  
       || (unsigned int)detid->subdetId() == StripSubdetector::TEC )  { 
    
    StripSubdetector StripDetId(*detid);


    if (accessMode_ == rphi ) {
    
      //return only r-phi RecHits; in case of double modules eliminate recurring r-phi RecHits
    
      if( !StripDetId.glued() ) {
	DetId useDetId(StripDetId.rawId());
	SiStripRecHit2DCollection::range rphiDetHits = rphiHits_->get(useDetId);
	for ( SiStripRecHit2DCollection::const_iterator rphiDetHit = rphiDetHits.first;
	      rphiDetHit != rphiDetHits.second; ++rphiDetHit ) {
	  RecHitVec.push_back((TrackingRecHit*)(&(*rphiDetHit)));
	}
      }
      
      if( StripDetId.glued() ) {
	DetId useDetId(StripDetId.glued());
	SiStripMatchedRecHit2DCollection::range matchedDetHits = matchedHits_->get(useDetId);

	for ( SiStripMatchedRecHit2DCollection::const_iterator matchedDetHit = matchedDetHits.first;
	      matchedDetHit != matchedDetHits.second; ++matchedDetHit ) {
	  //edm::LogError("RoadSearch") << "in LOOP " << (TrackingRecHit*)matchedDetHit->monoHit() ;//->geographicalId()).rawId(); 
	  std::vector<TrackingRecHit*>::const_iterator result = find(RecHitVec.begin(),RecHitVec.end(),(TrackingRecHit*)matchedDetHit->monoHit());
	  if( result==RecHitVec.end() ) {RecHitVec.push_back((TrackingRecHit*)matchedDetHit->monoHit());}
	}
      }
      
    }


    if (accessMode_ == standard ) {
      
      //return r-phi RecHits for single modules and matched RecHits for double modules

      if( !StripDetId.glued() ) {
	DetId useDetId(StripDetId.rawId());
	SiStripRecHit2DCollection::range rphiDetHits = rphiHits_->get(useDetId);
	for ( SiStripRecHit2DCollection::const_iterator rphiDetHit = rphiDetHits.first;
	      rphiDetHit != rphiDetHits.second; ++rphiDetHit ) {
	  RecHitVec.push_back((TrackingRecHit*)(&(*rphiDetHit)));
	}
      }

      if( StripDetId.glued() ) {
	DetId useDetId(StripDetId.glued());
	SiStripMatchedRecHit2DCollection::range matchedDetHits = matchedHits_->get(useDetId);
	
	for ( SiStripMatchedRecHit2DCollection::const_iterator matchedDetHit = matchedDetHits.first;
	      matchedDetHit != matchedDetHits.second; ++matchedDetHit ) {
	  RecHitVec.push_back((TrackingRecHit*)(&(*matchedDetHit)));
	}
      } 

    }
    
  } else if (    (unsigned int)detid->subdetId() == PixelSubdetector::PixelBarrel 
	      || (unsigned int)detid->subdetId() == PixelSubdetector::PixelEndcap) {
    
    SiPixelRecHitCollection::range pixelDetHits = pixelHits_->get(*detid);
    for ( SiPixelRecHitCollection::const_iterator pixelDetHit = pixelDetHits.first; 
	  pixelDetHit!= pixelDetHits.second; ++pixelDetHit) {
      
      RecHitVec.push_back((TrackingRecHit*)(&(*pixelDetHit)));
    }
    
  } else {

    edm::LogError("RoadSearch") << "NEITHER PIXEL NOR STRIP DETECTOR ID";

  }

  //edm::LogError("RoadSearch") << " DetHitAccess: size " << RecHitVec.size();
  //for (std::vector<TrackingRecHit*>::const_iterator testcn = RecHitVec.begin();
  //     testcn != RecHitVec.end(); ++testcn) {
  //edm::LogError("RoadSearch") << " DetHitAccess: " << (*testcn)->geographicalId().rawId();
  //}

  return RecHitVec;

  //Q for Oliver:
  //StripSubdetector::StripSubdetector(const DetId&)
  //StripSubdetector::StripSubdetector(detid) a;
  
}
  
