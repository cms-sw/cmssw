#ifndef RECOTRACKER_TRANSIENTTRACKINGRECHIT_TSiTrackerGSRecHit_H
#define RECOTRACKER_TRANSIENTTRACKINGRECHIT_TSiTrackerGSRecHit_H

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/Common/interface/RefGetter.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

class TSiTrackerGSRecHit : public TransientTrackingRecHit{
public:

  typedef SiTrackerGSRecHit2D::ClusterRef SiTrackerClusterRef;
  
  virtual ~TSiTrackerGSRecHit() {}

  virtual AlgebraicVector parameters() const {return theHitData.parameters();}
  virtual AlgebraicSymMatrix parametersError() const {
    return theHitData.parametersError();
  }
  
  virtual AlgebraicMatrix projectionMatrix() const {return theHitData.projectionMatrix();}
  virtual int dimension() const {return theHitData.dimension();}

  virtual LocalPoint localPosition() const {return theHitData.localPosition();}
  virtual LocalError localPositionError() const {return theHitData.localPositionError();}

  virtual const TrackingRecHit * hit() const {return &theHitData;};
  
  virtual std::vector<const TrackingRecHit*> recHits() const {
    return hit()->recHits();
  }
  virtual std::vector<TrackingRecHit*> recHits() {
    return theHitData.recHits();
  }

  virtual const GeomDetUnit* detUnit() const;

//  virtual bool canImproveWithTrack() const {return true;}
  virtual bool canImproveWithTrack() const {return false;}

  //RC virtual TSiStripRecHit2DLocalPos* clone(const TrajectoryStateOnSurface& ts) const;
  virtual RecHitPointer clone(const TrajectoryStateOnSurface& ts) const;

  // Extension of the TransientTrackingRecHit interface

  const SiTrackerGSRecHit2D* specificHit() const {return &theHitData;};

  static RecHitPointer build( const GeomDet * geom, const SiTrackerGSRecHit2D* rh,
			      float weight=1., float annealing=1.) {
    return RecHitPointer( new TSiTrackerGSRecHit( geom, rh, weight, annealing));
  }

  static RecHitPointer build( const LocalPoint& pos, const LocalError& err,
			      const GeomDet* det,
			      SiTrackerClusterRef const& clust,
			      const int& simhitId, const int& simtrackId,
                              const uint32_t& eeId, const int& pixelMultiplicityX,
                              const int& pixelMultiplicityY,
			      float weight=1., float annealing=1.) {
    return RecHitPointer( new TSiTrackerGSRecHit( pos, err, det, clust, simhitId, simtrackId, eeId, pixelMultiplicityX, pixelMultiplicityY, weight, annealing));
  }

private:

  SiTrackerGSRecHit2D               theHitData;

  TSiTrackerGSRecHit (const GeomDet * geom, const SiTrackerGSRecHit2D* rh,
                      float weight=1., float annealing=1.) : 
    TransientTrackingRecHit(geom, weight, annealing), theHitData(*rh)
  {}

  /// Creates the TrackingRecHit internally, avoids redundent cloning
  TSiTrackerGSRecHit( const LocalPoint& pos, const LocalError& err,
                      const GeomDet* det,
                      SiTrackerClusterRef const& clust,
                      const int& simhitId,
                      const int& simtrackId, const uint32_t& eeId,
                      const int& pixelMultiplicityX, const int& pixelMultiplicityY,
                      float weight=1., float annealing=1.) :
    TransientTrackingRecHit(det, weight, annealing), theHitData(pos, err, det->geographicalId(), simhitId, simtrackId, eeId, clust, pixelMultiplicityX, pixelMultiplicityY) {} 

/*   TSiTrackerGSRecHit( const LocalPoint& pos, const LocalError& err, */
/*                       const GeomDet* det, */
/*                       SiTrackerClusterRef const& clust, */
/*                       const int& simhitId, */
/*                       const int& simtrackId, const uint32_t& eeId, */
/*                       const int& pixelMultiplicityX, const int& pixelMultiplicityY, */
/*                       float weight, float annealing) : */
/*     TransientTrackingRecHit(det, weight, annealing), theHitData(pos, err, det->geographicalId(), simhitId, simtrackId, eeId, clust, pixelMultiplicityX, pixelMultiplicityY) {}  */
  
  

  virtual TSiTrackerGSRecHit* clone() const {
    return new TSiTrackerGSRecHit(*this);
  }

};

#endif
