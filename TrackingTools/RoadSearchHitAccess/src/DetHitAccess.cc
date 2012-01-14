#include "TrackingTools/RoadSearchHitAccess/interface/DetHitAccess.h"

/// I need this because DetHitAccess assumes that it can search a hit container using a detid which is not there
template<typename T>
inline edmNew::DetSet<T> detSetOrEmpty(const edmNew::DetSetVector<T> &dsv, DetId detid) {
    typename edmNew::DetSetVector<T>::const_iterator iter = dsv.find(detid.rawId());
    if (iter == dsv.end()) {
        return typename edmNew::DetSet<T>(detid.rawId(), static_cast<const T *>(0), size_t(0) );
    } else {
        return *iter;
    }
}


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
	  SiStripRecHit2DCollection::DetSet rphiDetHits = detSetOrEmpty(*rphiHits_, *detid);
	  for ( SiStripRecHit2DCollection::DetSet::const_iterator rphiDetHit = rphiDetHits.begin();
		rphiDetHit != rphiDetHits.end(); 
		++rphiDetHit ) {
	    RecHitVec.push_back((TrackingRecHit*)(&(*rphiDetHit)));
	  }
	} else {
	  edm::LogWarning("RoadSearch") << "rphi RecHit collection not set properly";
	}
      } else {
	/* 
	 * VI January 2012 not supported anymore (at least in this form)
         * and (by-the-way) is infinitely slow with such a double loop!!!!!
         *
	if ( matchedHits_ != 0 ) {
	  DetId useDetId(StripDetId.glued());
	  SiStripMatchedRecHit2DCollection::DetSet matchedDetHits = detSetOrEmpty(*matchedHits_, useDetId);
	  for ( SiStripMatchedRecHit2DCollection::DetSet::const_iterator matchedDetHit = matchedDetHits.begin();
		matchedDetHit != matchedDetHits.end(); ++matchedDetHit ) {
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
        */
      }
    } else if (accessMode_ == rphi_stereo ) {
      //
      //return only r-phi and stereo RecHits
      //
      if( !StripDetId.glued() ) {
	if ( rphiHits_ != 0 ) {
	  SiStripRecHit2DCollection::DetSet rphiDetHits = detSetOrEmpty(*rphiHits_, *detid);
	  for ( SiStripRecHit2DCollection::DetSet::const_iterator rphiDetHit = rphiDetHits.begin();
		rphiDetHit != rphiDetHits.end(); 
		++rphiDetHit ) {
	    RecHitVec.push_back((TrackingRecHit*)(&(*rphiDetHit)));
	  }
	} else {
	  edm::LogWarning("RoadSearch") << "rphi RecHit collection not set properly";
	}
	
      } else {
	DetId rphiDetId(StripDetId.glued()+2);
	if ( rphiHits_ != 0 ) {
	  SiStripRecHit2DCollection::DetSet rphiDetHits = detSetOrEmpty(*rphiHits_, rphiDetId);
	  for ( SiStripRecHit2DCollection::DetSet::const_iterator rphiDetHit = rphiDetHits.begin();
		rphiDetHit != rphiDetHits.end(); 
		++rphiDetHit ) {
	    RecHitVec.push_back((TrackingRecHit*)(&(*rphiDetHit)));
	  }
	} else {
	  edm::LogWarning("RoadSearch") << "rphi RecHit collection not set properly";
	}

	DetId stereoDetId(StripDetId.glued()+1);
	if ( stereoHits_ != 0 ) {
	  SiStripRecHit2DCollection::DetSet stereoDetHits = detSetOrEmpty(*stereoHits_, stereoDetId);
	  for ( SiStripRecHit2DCollection::DetSet::const_iterator stereoDetHit = stereoDetHits.begin();
		stereoDetHit != stereoDetHits.end(); 
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
	  SiStripRecHit2DCollection::DetSet rphiDetHits = detSetOrEmpty(*rphiHits_, *detid);
	  for ( SiStripRecHit2DCollection::DetSet::const_iterator rphiDetHit = rphiDetHits.begin();
		rphiDetHit != rphiDetHits.end(); 
		++rphiDetHit ) {
	    RecHitVec.push_back((TrackingRecHit*)(&(*rphiDetHit)));
	  }
	} else {
	  edm::LogWarning("RoadSearch") << "rphi RecHit collection not set properly";
	}
      } else {
	if ( matchedHits_ != 0 ) {
	  DetId useDetId(StripDetId.glued());
	  SiStripMatchedRecHit2DCollection::DetSet matchedDetHits = detSetOrEmpty(*matchedHits_, useDetId);
	  for ( SiStripMatchedRecHit2DCollection::DetSet::const_iterator matchedDetHit = matchedDetHits.begin();
		matchedDetHit != matchedDetHits.end(); ++matchedDetHit ) {
	    RecHitVec.push_back((TrackingRecHit*)(&(*matchedDetHit)));
	  }
	} else {
	  edm::LogWarning("RoadSearch") << "matched RecHit collection not set properly";
	}
	
	//check for additional r-phi RecHits (not used by matched RecHits)
	if(use_rphiRecHits_) {
	  DetId rphiDetId(StripDetId.glued()+2);
	  if ( rphiHits_ != 0 ) {
	    SiStripRecHit2DCollection::DetSet rphiDetHits = detSetOrEmpty(*rphiHits_, rphiDetId);
	    for ( SiStripRecHit2DCollection::DetSet::const_iterator rphiDetHit = rphiDetHits.begin();
		  rphiDetHit != rphiDetHits.end(); ++rphiDetHit ) {
	      bool use_rphi=true;
	      DetId useDetId(StripDetId.glued());
	      SiStripMatchedRecHit2DCollection::DetSet matchedDetHits = detSetOrEmpty(*matchedHits_, useDetId);
	      //SiStripMatchedRecHit2DCollection::DetSet matchedDetHits = detSetOrEmpty(*matchedHits_, *detid);
	      for ( SiStripMatchedRecHit2DCollection::DetSet::const_iterator matchedDetHit = matchedDetHits.begin();
		    matchedDetHit != matchedDetHits.end(); ++matchedDetHit ) { 
		if (rphiDetHit->sharesInput((TrackingRecHit*)(&(*matchedDetHit)), TrackingRecHit::some)) {
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
	    SiStripRecHit2DCollection::DetSet stereoDetHits = detSetOrEmpty(*stereoHits_, stereoDetId);
	    for ( SiStripRecHit2DCollection::DetSet::const_iterator stereoDetHit = stereoDetHits.begin();
		  stereoDetHit != stereoDetHits.end(); ++stereoDetHit ) {
	      bool use_stereo=true;
	      DetId useDetId(StripDetId.glued());
	      SiStripMatchedRecHit2DCollection::DetSet matchedDetHits = detSetOrEmpty(*matchedHits_, useDetId);
	      //SiStripMatchedRecHit2DCollection::DetSet matchedDetHits = detSetOrEmpty(*matchedHits_, *detid);
	      for ( SiStripMatchedRecHit2DCollection::DetSet::const_iterator matchedDetHit = matchedDetHits.begin();
		    matchedDetHit != matchedDetHits.end(); ++matchedDetHit ) { 
		if (stereoDetHit->sharesInput((TrackingRecHit*)(&(*matchedDetHit)), TrackingRecHit::some)) {
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
      SiPixelRecHitCollection::DetSet pixelDetHits = detSetOrEmpty(*pixelHits_, *detid);
      for ( SiPixelRecHitCollection::DetSet::const_iterator pixelDetHit = pixelDetHits.begin(); 
	    pixelDetHit!= pixelDetHits.end(); ++pixelDetHit) {
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

