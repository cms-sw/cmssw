#include "TrackingTools/RoadSearchHitAccess/interface/DetHitAccess.h"

DetHitAccess::DetHitAccess(){

  // default for access mode
  accessMode_ = standard;

};

DetHitAccess::~DetHitAccess(){
};

DetHitAccess::DetHitAccess(const SiStripRecHit2DCollection* rphiRecHits,
			   const SiStripRecHit2DCollection* stereoRecHits,
			   const SiStripMatchedRecHit2DCollection* matchedRecHits,
			   const SiPixelRecHitCollection* pixelRecHits) {
  
  // default for access mode
  accessMode_ = standard;

  // set collections
  setCollections(rphiRecHits,stereoRecHits,matchedRecHits,pixelRecHits);
  
}
		
void DetHitAccess::setCollections(const SiStripRecHit2DCollection* rphiRecHits,
				  const SiStripRecHit2DCollection* stereoRecHits,
				  const SiStripMatchedRecHit2DCollection* matchedRecHits,
				  const SiPixelRecHitCollection* pixelRecHits) {
  
  rphiHits_ = rphiRecHits;
  stereoHits_ = stereoRecHits;
  matchedHits_ = matchedRecHits;
  pixelHits_ = pixelRecHits;

  matchedHitsDetIds_ = matchedRecHits->ids();
  rphiHitsDetIds_    = rphiRecHits->ids();
  stereoHitsDetIds_  = stereoRecHits->ids();
  pixelHitsDetIds_   = pixelRecHits->ids();
  
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
	try {
	  if ( rphiHitsDetIds_.end() != std::find(rphiHitsDetIds_.begin(),rphiHitsDetIds_.end(),useDetId) ) {
	    SiStripRecHit2DCollection::range rphiDetHits = rphiHits_->get(useDetId);
	    for ( SiStripRecHit2DCollection::const_iterator rphiDetHit = rphiDetHits.first;
		  rphiDetHit != rphiDetHits.second; 
		  ++rphiDetHit ) {
	      RecHitVec.push_back((TrackingRecHit*)(&(*rphiDetHit)));
	    }
	  }
	} catch(const std::exception& er) {
	  edm::LogWarning("RoadSearch") << "rphi RecHit collection not set properly";
	}
      } else {
	DetId useDetId(StripDetId.glued());
	try {
	  if ( matchedHitsDetIds_.end() != std::find(matchedHitsDetIds_.begin(),matchedHitsDetIds_.end(),useDetId) ) {
	    SiStripMatchedRecHit2DCollection::range matchedDetHits = matchedHits_->get(useDetId);
	    for ( SiStripMatchedRecHit2DCollection::const_iterator matchedDetHit = matchedDetHits.first;
		  matchedDetHit != matchedDetHits.second; ++matchedDetHit ) {
	      bool add = true;
	      TrackingRecHit *rphi = (TrackingRecHit*)matchedDetHit->monoHit();
	      for ( std::vector<TrackingRecHit*>::iterator hit = RecHitVec.begin();
		    hit != RecHitVec.end();
		    ++hit ) {
		if ((*hit)->localPosition().x() == rphi->localPosition().x()) {
		  if ((*hit)->localPosition().y() == rphi->localPosition().y()) {
		    add = false;
		    break;
		  }
		}
	      }
	      if ( add ) {
		RecHitVec.push_back(rphi);
	      }
	    }
	  }
	} catch(const std::exception& er) {
	  edm::LogWarning("RoadSearch") << "matched RecHit collection not set properly";
	}
      }
    } else if (accessMode_ == standard ) {
      
      //single modules: return r-phi RecHits
      //double modules: return matched RecHits + r-phi RecHits that are not used by matched RecHits

      if( !StripDetId.glued() ) {
	DetId useDetId(StripDetId.rawId());
	try {
	  if ( rphiHitsDetIds_.end() != std::find(rphiHitsDetIds_.begin(),rphiHitsDetIds_.end(),useDetId) ) {
	    SiStripRecHit2DCollection::range rphiDetHits = rphiHits_->get(useDetId);
	    for ( SiStripRecHit2DCollection::const_iterator rphiDetHit = rphiDetHits.first;
		  rphiDetHit != rphiDetHits.second; 
		  ++rphiDetHit ) {
	      RecHitVec.push_back((TrackingRecHit*)(&(*rphiDetHit)));
	    }
	  }
	} catch(const std::exception& er) {
	  edm::LogWarning("RoadSearch") << "rphi RecHit collection not set properly";
	}
      } else {
	DetId useDetId(StripDetId.glued());
	try {
	  if ( matchedHitsDetIds_.end() != std::find(matchedHitsDetIds_.begin(),matchedHitsDetIds_.end(),useDetId) ) {
	    SiStripMatchedRecHit2DCollection::range matchedDetHits = matchedHits_->get(useDetId);
	    for ( SiStripMatchedRecHit2DCollection::const_iterator matchedDetHit = matchedDetHits.first;
		  matchedDetHit != matchedDetHits.second; ++matchedDetHit ) {
	      RecHitVec.push_back((TrackingRecHit*)(&(*matchedDetHit)));
	    }
	  }
	} catch(const std::exception& er) {
	  edm::LogWarning("RoadSearch") << "matched RecHit collection not set properly";
	}
	
	//check for additional r-phi RecHits (not used by matched RecHits)
	if(use_rphiRecHits_) {
	  DetId rphiDetId(StripDetId.glued()+2);
	  try {
	    if ( rphiHitsDetIds_.end() != std::find(rphiHitsDetIds_.begin(),rphiHitsDetIds_.end(),rphiDetId) ) {
	      SiStripRecHit2DCollection::range rphiDetHits = rphiHits_->get(rphiDetId);
	      for ( SiStripRecHit2DCollection::const_iterator rphiDetHit = rphiDetHits.first;
		    rphiDetHit != rphiDetHits.second; ++rphiDetHit ) {
		bool use_rphi=true;
		SiStripMatchedRecHit2DCollection::range matchedDetHits = matchedHits_->get(useDetId);
		for ( SiStripMatchedRecHit2DCollection::const_iterator matchedDetHit = matchedDetHits.first;
		      matchedDetHit != matchedDetHits.second; ++matchedDetHit ) { 
		  if (rphiDetHit->localPosition().x()==matchedDetHit->monoHit()->localPosition().x() 
		      && rphiDetHit->localPosition().y()==matchedDetHit->monoHit()->localPosition().y() ) {
		    use_rphi=false;
		    break;
		  }
		}
		if(use_rphi) RecHitVec.push_back((TrackingRecHit*)(&(*rphiDetHit)));
	      }
	    }
	  } catch(const std::exception& er) {
	    edm::LogWarning("RoadSearch") << "rphi RecHit collection not set properly";
	  }
	}
	  
	//check for additional stereo RecHits (not used by matched RecHits)
	if(use_stereoRecHits_) {
	  DetId stereoDetId(StripDetId.glued()+1);
	  try {
	    if ( stereoHitsDetIds_.end() != std::find(stereoHitsDetIds_.begin(),stereoHitsDetIds_.end(),stereoDetId) ) {
	      SiStripRecHit2DCollection::range stereoDetHits = stereoHits_->get(stereoDetId);
	      for ( SiStripRecHit2DCollection::const_iterator stereoDetHit = stereoDetHits.first;
		    stereoDetHit != stereoDetHits.second; ++stereoDetHit ) {
		bool use_stereo=true;
		SiStripMatchedRecHit2DCollection::range matchedDetHits = matchedHits_->get(useDetId);
		for ( SiStripMatchedRecHit2DCollection::const_iterator matchedDetHit = matchedDetHits.first;
		      matchedDetHit != matchedDetHits.second; ++matchedDetHit ) { 
		  if (stereoDetHit->localPosition().x()==matchedDetHit->monoHit()->localPosition().x() 
		      && stereoDetHit->localPosition().y()==matchedDetHit->monoHit()->localPosition().y() ) {
		    use_stereo=false;
		    break;
		  }
		}
		if(use_stereo) RecHitVec.push_back((TrackingRecHit*)(&(*stereoDetHit)));
	      }
	    }
	  } catch(const std::exception& er) {
	    edm::LogWarning("RoadSearch") << "stereo RecHit collection not set properly";
	  }  
	}
      }
    }
    
  } else if (    (unsigned int)detid->subdetId() == PixelSubdetector::PixelBarrel 
		 || (unsigned int)detid->subdetId() == PixelSubdetector::PixelEndcap) {
    
    try {
      if ( pixelHitsDetIds_.end() != std::find(pixelHitsDetIds_.begin(),pixelHitsDetIds_.end(),*detid) ) {
	SiPixelRecHitCollection::range pixelDetHits = pixelHits_->get(*detid);
	for ( SiPixelRecHitCollection::const_iterator pixelDetHit = pixelDetHits.first; 
	      pixelDetHit!= pixelDetHits.second; ++pixelDetHit) {
	  RecHitVec.push_back((TrackingRecHit*)(&(*pixelDetHit)));
	}
      }
    } catch(const std::exception& er) {
      edm::LogWarning("RoadSearch") << "pixel RecHit collection not set properly";
    } 
  } else {

    edm::LogError("RoadSearch") << "NEITHER PIXEL NOR STRIP DETECTOR ID";

  }


  return RecHitVec;
}
  
