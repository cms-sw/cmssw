#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiPixelRecHit_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiPixelRecHit_H

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"

class GeomDetUnit;

class TSiPixelRecHit : public TransientTrackingRecHit {
public:

  /// This constructor clones the TrackingRecHit, it should be used when the 
  /// TrackingRecHit exist already in some collection
  TSiPixelRecHit(const GeomDet * geom, const SiPixelRecHit* rh, 
		 const PixelClusterParameterEstimator* cpe) : 
    TransientTrackingRecHit(geom), theHitData(rh->clone()), theCPE(cpe) {}

  /// Creates the TrackingRecHit internally, avoids redundent cloning
  TSiPixelRecHit( const LocalPoint& pos, const LocalError& err,
		  const GeomDet* det, 
		  const SiPixelCluster& cluster,
		  const PixelClusterParameterEstimator* cpe);

  TSiPixelRecHit( const TSiPixelRecHit& other ) :
    TransientTrackingRecHit( other.det()), 
    theHitData( other.specificHit()->clone()),
    theCPE( other.cpe())  {}

  virtual ~TSiPixelRecHit() {delete theHitData;}

  virtual TSiPixelRecHit * clone() const {
    return new TSiPixelRecHit(*this);
  }

  virtual AlgebraicVector parameters() const {return theHitData->parameters();}
  virtual AlgebraicSymMatrix parametersError() const {return theHitData->parametersError();}
  virtual DetId geographicalId() const {return theHitData->geographicalId();}
  virtual AlgebraicMatrix projectionMatrix() const {return theHitData->projectionMatrix();}
  virtual int dimension() const {return theHitData->dimension();}

  virtual LocalPoint localPosition() const {return theHitData->localPosition();}
  virtual LocalError localPositionError() const {return theHitData->localPositionError();}

  virtual const TrackingRecHit * hit() const {return theHitData;};
  
  virtual bool isValid() const {return theHitData->isValid();}

  virtual std::vector<const TrackingRecHit*> recHits() const {
    return hit()->recHits();
  }
  virtual std::vector<TrackingRecHit*> recHits() {
    return theHitData->recHits();
  }

  virtual bool canImproveWithTrack() const {return true;}

  virtual TSiPixelRecHit* clone (const TrajectoryStateOnSurface& ts) const;

  virtual const GeomDetUnit* detUnit() const;

  // Extension of the TransientTrackingRecHit interface

  const SiPixelRecHit* specificHit() const {return theHitData;};
  const PixelClusterParameterEstimator* cpe() const {return theCPE;}

private:

  SiPixelRecHit*                        theHitData;
  const PixelClusterParameterEstimator* theCPE;

};



#endif
