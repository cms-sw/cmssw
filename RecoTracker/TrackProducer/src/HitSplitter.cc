#include "RecoTracker/TrackProducer/interface/HitSplitter.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"



void HitSplitter::splitHits(edm::OwnVector<TrackingRecHit>::const_iterator beginInput,
			    edm::OwnVector<TrackingRecHit>::const_iterator endInput,
			    const TransientTrackingRecHitBuilder* builder,
			    const TrackingGeometry * theG,
			    bool reverse,
			    TransientTrackingRecHit::RecHitContainer& outputCollection,
			    float& ndof) {
  
  bool isInOut = true; // TOBE FIXED: this should be taken as function argument
  int order = isInOut ? 1 : -1;
  // ----

  if(!reverse){
    for (edm::OwnVector<TrackingRecHit>::const_iterator hit=beginInput;
       hit!=endInput; ++hit){
      doSplit( &(*hit),builder,theG,order,outputCollection,ndof);
    }
  }else{
    for (edm::OwnVector<TrackingRecHit>::const_iterator hit=endInput-1;
       hit!=beginInput-1; --hit){
      doSplit( &(*hit),builder,theG,order,outputCollection,ndof);
    }
  }
}






  void HitSplitter::splitHits(trackingRecHit_iterator beginInput,
			      trackingRecHit_iterator endInput,
			      const TransientTrackingRecHitBuilder* builder,
			      const TrackingGeometry * theG,
			      bool reverse,
			      TransientTrackingRecHit::RecHitContainer& outputCollection,
			      float& ndof){

  bool isInOut = true; // TOBE FIXED: this should be taken as function argument
  int order = isInOut ? 1 : -1;
  // ----
  
  if(!reverse){
    for (trackingRecHit_iterator hit = beginInput; hit!=endInput; ++hit){    
      doSplit( &(**hit),builder,theG,order,outputCollection,ndof);
    }
  }else{
    for (trackingRecHit_iterator hit = endInput-1; hit!=beginInput-1; --hit){    
      doSplit( &(**hit),builder,theG,order,outputCollection,ndof);
    }
  }

  

}



void HitSplitter::doSplit(const TrackingRecHit* hit,
			  const TransientTrackingRecHitBuilder* builder,
			  const TrackingGeometry * theG,
			  int order,
			  TransientTrackingRecHit::RecHitContainer& outputCollection,
			  float& ndof){

  // --- WARNING
  // this method to decide the order of the split hits works only for
  // LHC collsion track reconstructed in-out or out-in. 
  // It doesn't apply to cosmic tracks going out-in (in Y+ tracker) and then in-out 
  // (in Y- tracker).  For the same reason it does not work for beam-halo muon
  // More general implementation has to be provided

  const SiStripMatchedRecHit2D* matched = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);
  if(matched){	      
    DetId id = matched->geographicalId();
    uint32_t subdet = id.subdetId();
    if( (subdet == StripSubdetector::TIB) || (subdet == StripSubdetector::TOB) ){	      
      const GeomDet* tmpDet;  
	
      tmpDet = theG->idToDet(matched->monoHit()->geographicalId());	      
      double monoR = tmpDet->surface().position().perp();
      
      tmpDet  = theG->idToDet(matched->stereoHit()->geographicalId());	      
      double stereoR = tmpDet->surface().position().perp();
	      
      if( (monoR - stereoR)*order < 0 ){
	outputCollection.push_back(builder->build(matched->monoHit()));
	outputCollection.push_back(builder->build(matched->stereoHit()));
	ndof += 
	  (matched->monoHit()->dimension())*(matched->monoHit()->weight());
	ndof += 
	  (matched->stereoHit()->dimension())*(matched->stereoHit()->weight());
      }else{
	outputCollection.push_back(builder->build(matched->stereoHit()));
	outputCollection.push_back(builder->build(matched->monoHit()));
	ndof += 
	  (matched->monoHit()->dimension())*(matched->monoHit()->weight());
	ndof +=
	  (matched->stereoHit()->dimension())*(matched->stereoHit()->weight());
      }
    }else if( (subdet == StripSubdetector::TID) || (subdet == StripSubdetector::TEC) ){
      const GeomDet* tmpDet;  
      
      tmpDet = theG->idToDet(matched->monoHit()->geographicalId());	      
      double monoZ = tmpDet->surface().position().z();
      
      tmpDet  = theG->idToDet(matched->stereoHit()->geographicalId());	      
      double stereoZ = tmpDet->surface().position().z();
      
      if( (fabs(monoZ)- fabs(stereoZ))*order < 0 ){
	outputCollection.push_back(builder->build(matched->monoHit()));
	outputCollection.push_back(builder->build(matched->stereoHit()));
	ndof += 
	  (matched->monoHit()->dimension())*(matched->monoHit()->weight());
	ndof += 
	    (matched->stereoHit()->dimension())*(matched->stereoHit()->weight());
      }else{
	outputCollection.push_back(builder->build(matched->stereoHit()));
	outputCollection.push_back(builder->build(matched->monoHit()));
	ndof += 
	  (matched->monoHit()->dimension())*(matched->monoHit()->weight());
	ndof +=
	  (matched->stereoHit()->dimension())*(matched->stereoHit()->weight());
      }
    }
    return;
  }
  
  const ProjectedSiStripRecHit2D* projected = dynamic_cast<const ProjectedSiStripRecHit2D*>(hit);
  if(projected) {
    outputCollection.push_back(builder->build( projected->originalHit().clone() ));
    ndof += (projected->dimension())*(projected->weight());
    return;
  }
  
  outputCollection.push_back(builder->build(hit ));
  if (hit->isValid()){
    //std::cout << "++++ hit global R: " 
    //<< outputCollection.back()->globalPosition().perp() << std::endl;
    ndof += (hit->dimension())*(hit->weight());
  }	    
}
















