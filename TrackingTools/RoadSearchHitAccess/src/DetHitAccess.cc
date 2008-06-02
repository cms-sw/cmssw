#include "TrackingTools/RoadSearchHitAccess/interface/DetHitAccess.h"

DetHitAccess::DetHitAccess() {
  // default for access mode
  accessMode_ = standard;

  use_rphiRecHits_ = true;
  use_rphiRecHits_ = true;
}

DetHitAccess::~DetHitAccess(){
}

DetHitAccess::DetHitAccess(const SiStripRecHit2DCollection* rphiRecHits,
			   const SiStripRecHit2DCollection* stereoRecHits,
			   const SiStripMatchedRecHit2DCollection* matchedRecHits,
			   const SiPixelRecHitCollection* pixelRecHits) {
  
  // default for access mode
  accessMode_ = standard;

  use_rphiRecHits_   = true;
  use_stereoRecHits_ = true;

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

}


std::vector<TrackingRecHit*> DetHitAccess::getHitVector(const DetId* detid) {
  //
  //DetId that is given to getHitVector is *always* rphi
  //
  
  std::vector<TrackingRecHit*> RecHitVec;
  
  if ( (unsigned int)detid->subdetId() == StripSubdetector::TIB  
       || (unsigned int)detid->subdetId() == StripSubdetector::TOB  
       || (unsigned int)detid->subdetId() == StripSubdetector::TID  
       || (unsigned int)detid->subdetId() == StripSubdetector::TEC )  { 
    
    StripSubdetector StripDetId(*detid);


    if (accessMode_ == rphi ) {
      //
      //return only r-phi RecHits; in case of double modules eliminate recurring r-phi RecHits
      //
      if( !StripDetId.glued() ) {
	if ( rphiHits_ != 0 ) {
	  SiStripRecHit2DCollection::range rphiDetHits = rphiHits_->get(*detid);
	  for ( SiStripRecHit2DCollection::const_iterator rphiDetHit = rphiDetHits.first;
		rphiDetHit != rphiDetHits.second; 
		++rphiDetHit ) {
	    RecHitVec.push_back((TrackingRecHit*)(&(*rphiDetHit)));
	  }
	} else {
	  edm::LogWarning("RoadSearch") << "rphi RecHit collection not set properly";
	}
      } else {
	if ( matchedHits_ != 0 ) {
	  DetId useDetId(StripDetId.glued());
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
	} else {
	  edm::LogWarning("RoadSearch") << "matched RecHit collection not set properly";
	}
      }
    } else if (accessMode_ == rphi_stereo ) {
      //
      //return only r-phi and stereo RecHits
      //
      if( !StripDetId.glued() ) {
	if ( rphiHits_ != 0 ) {
	  SiStripRecHit2DCollection::range rphiDetHits = rphiHits_->get(*detid);
	  for ( SiStripRecHit2DCollection::const_iterator rphiDetHit = rphiDetHits.first;
		rphiDetHit != rphiDetHits.second; 
		++rphiDetHit ) {
	    RecHitVec.push_back((TrackingRecHit*)(&(*rphiDetHit)));
	  }
	} else {
	  edm::LogWarning("RoadSearch") << "rphi RecHit collection not set properly";
	}
	
      } else {
	DetId rphiDetId(StripDetId.glued()+2);
	if ( rphiHits_ != 0 ) {
	  SiStripRecHit2DCollection::range rphiDetHits = rphiHits_->get(rphiDetId);
	  for ( SiStripRecHit2DCollection::const_iterator rphiDetHit = rphiDetHits.first;
		rphiDetHit != rphiDetHits.second; 
		++rphiDetHit ) {
	    RecHitVec.push_back((TrackingRecHit*)(&(*rphiDetHit)));
	  }
	} else {
	  edm::LogWarning("RoadSearch") << "rphi RecHit collection not set properly";
	}

	DetId stereoDetId(StripDetId.glued()+1);
	if ( stereoHits_ != 0 ) {
	  SiStripRecHit2DCollection::range stereoDetHits = stereoHits_->get(stereoDetId);
	  for ( SiStripRecHit2DCollection::const_iterator stereoDetHit = stereoDetHits.first;
		stereoDetHit != stereoDetHits.second; 
		++stereoDetHit ) {
	    RecHitVec.push_back((TrackingRecHit*)(&(*stereoDetHit)));
	  }
	} else {
	  edm::LogWarning("RoadSearch") << "stereo RecHit collection not set properly";
	}
      }
    } else if (accessMode_ == standard ) {
      //
      //single modules: return r-phi RecHits
      //double modules: return matched RecHits + r-phi RecHits that are not used by matched RecHits
      //
      if( !StripDetId.glued() ) {
	if ( rphiHits_ != 0 ) {
	  SiStripRecHit2DCollection::range rphiDetHits = rphiHits_->get(*detid);
	  for ( SiStripRecHit2DCollection::const_iterator rphiDetHit = rphiDetHits.first;
		rphiDetHit != rphiDetHits.second; 
		++rphiDetHit ) {
	    RecHitVec.push_back((TrackingRecHit*)(&(*rphiDetHit)));
	  }
	} else {
	  edm::LogWarning("RoadSearch") << "rphi RecHit collection not set properly";
	}
      } else {
	if ( matchedHits_ != 0 ) {
	  DetId useDetId(StripDetId.glued());
	  SiStripMatchedRecHit2DCollection::range matchedDetHits = matchedHits_->get(useDetId);
	  for ( SiStripMatchedRecHit2DCollection::const_iterator matchedDetHit = matchedDetHits.first;
		matchedDetHit != matchedDetHits.second; ++matchedDetHit ) {
	    RecHitVec.push_back((TrackingRecHit*)(&(*matchedDetHit)));
	  }
	} else {
	  edm::LogWarning("RoadSearch") << "matched RecHit collection not set properly";
	}
	
	//check for additional r-phi RecHits (not used by matched RecHits)
	if(use_rphiRecHits_) {
	  DetId rphiDetId(StripDetId.glued()+2);
	  if ( rphiHits_ != 0 ) {
	    SiStripRecHit2DCollection::range rphiDetHits = rphiHits_->get(rphiDetId);
	    for ( SiStripRecHit2DCollection::const_iterator rphiDetHit = rphiDetHits.first;
		  rphiDetHit != rphiDetHits.second; ++rphiDetHit ) {
	      bool use_rphi=true;
	      DetId useDetId(StripDetId.glued());
	      SiStripMatchedRecHit2DCollection::range matchedDetHits = matchedHits_->get(useDetId);
	      //SiStripMatchedRecHit2DCollection::range matchedDetHits = matchedHits_->get(*detid);
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
	  } else {
	    edm::LogWarning("RoadSearch") << "rphi RecHit collection not set properly";
	  }
	}
	  
	//check for additional stereo RecHits (not used by matched RecHits)
	if(use_stereoRecHits_) {
	  DetId stereoDetId(StripDetId.glued()+1);
	  if ( stereoHits_ != 0 ) {
	    SiStripRecHit2DCollection::range stereoDetHits = stereoHits_->get(stereoDetId);
	    for ( SiStripRecHit2DCollection::const_iterator stereoDetHit = stereoDetHits.first;
		  stereoDetHit != stereoDetHits.second; ++stereoDetHit ) {
	      bool use_stereo=true;
	      DetId useDetId(StripDetId.glued());
	      SiStripMatchedRecHit2DCollection::range matchedDetHits = matchedHits_->get(useDetId);
	      //SiStripMatchedRecHit2DCollection::range matchedDetHits = matchedHits_->get(*detid);
	      for ( SiStripMatchedRecHit2DCollection::const_iterator matchedDetHit = matchedDetHits.first;
		    matchedDetHit != matchedDetHits.second; ++matchedDetHit ) { 
		if (stereoDetHit->localPosition().x()==matchedDetHit->stereoHit()->localPosition().x() 
		    && stereoDetHit->localPosition().y()==matchedDetHit->stereoHit()->localPosition().y() ) {
		  use_stereo=false;
		  break;
		}
	      }
	      if(use_stereo) RecHitVec.push_back((TrackingRecHit*)(&(*stereoDetHit)));
	    }
	  } else {
	    edm::LogWarning("RoadSearch") << "stereo RecHit collection not set properly";
	  }  
	}
      }
    }
    
  } else if (    (unsigned int)detid->subdetId() == PixelSubdetector::PixelBarrel 
		 || (unsigned int)detid->subdetId() == PixelSubdetector::PixelEndcap) {
    
    if ( pixelHits_ != 0 ) {
      SiPixelRecHitCollection::range pixelDetHits = pixelHits_->get(*detid);
      for ( SiPixelRecHitCollection::const_iterator pixelDetHit = pixelDetHits.first; 
	    pixelDetHit!= pixelDetHits.second; ++pixelDetHit) {
	RecHitVec.push_back((TrackingRecHit*)(&(*pixelDetHit)));
      }
    } else {
      edm::LogWarning("RoadSearch") << "pixel RecHit collection not set properly";
    } 
  } else {
    
    edm::LogError("RoadSearch") << "NEITHER PIXEL NOR STRIP DETECTOR ID";

  }
  
  
  return RecHitVec;
}

