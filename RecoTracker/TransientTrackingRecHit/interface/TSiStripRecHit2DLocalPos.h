#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripRecHit2DLocalPos_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripRecHit2DLocalPos_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/HelpertRecHit2DLocalPos.h"

class TSiStripRecHit2DLocalPos : public TransientTrackingRecHit{
public:


//  TSiStripRecHit2DLocalPos (const GeomDet * geom, const SiStripRecHit2D* rh,
//			    const StripClusterParameterEstimator* cpe) : 
//    TransientTrackingRecHit(geom), theHitData(rh->clone()), theCPE(cpe) 
//  {}

//  /// Creates the TrackingRecHit internally, avoids redundent cloning
//  TSiStripRecHit2DLocalPos( const LocalPoint& pos, const LocalError& err,
//			    const GeomDet* det,
//			    const edm::Ref< edm::DetSetVector<SiStripCluster>,SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster>  >  clust,
//			    const StripClusterParameterEstimator* cpe);

//  TSiStripRecHit2DLocalPos( const TSiStripRecHit2DLocalPos& other ) :
//    TransientTrackingRecHit( other.det()), 
//    theHitData( other.specificHit()->clone()),
//    theCPE( other.cpe()) {}


  virtual ~TSiStripRecHit2DLocalPos() {delete theHitData;}

//   virtual TSiStripRecHit2DLocalPos* clone() const {
//     return new TSiStripRecHit2DLocalPos(*this);
//   }

  virtual AlgebraicVector parameters() const {return theHitData->parameters();}
  virtual AlgebraicSymMatrix parametersError() const {
    return HelpertRecHit2DLocalPos().parError( theHitData->localPositionError(), *det()); 
    //    return theHitData->parametersError();
  }
  
  virtual DetId geographicalId() const {return theHitData->geographicalId();}
  virtual AlgebraicMatrix projectionMatrix() const {return theHitData->projectionMatrix();}
  virtual int dimension() const {return theHitData->dimension();}

  virtual LocalPoint localPosition() const {return theHitData->localPosition();}
  virtual LocalError localPositionError() const {return theHitData->localPositionError();}

  virtual const TrackingRecHit * hit() const {return theHitData;};
  
  virtual bool isValid() const{return theHitData->isValid();}

  virtual std::vector<const TrackingRecHit*> recHits() const {
    return hit()->recHits();
  }
  virtual std::vector<TrackingRecHit*> recHits() {
    return theHitData->recHits();
  }

  virtual const GeomDetUnit* detUnit() const;

  virtual bool canImproveWithTrack() const {return true;}

  //RC virtual TSiStripRecHit2DLocalPos* clone(const TrajectoryStateOnSurface& ts) const;
  virtual RecHitPointer clone(const TrajectoryStateOnSurface& ts) const;

  // Extension of the TransientTrackingRecHit interface

  const SiStripRecHit2D* specificHit() const {return theHitData;};
  const StripClusterParameterEstimator* cpe() const {return theCPE;}

  static RecHitPointer build( const GeomDet * geom, const SiStripRecHit2D* rh,
			      const StripClusterParameterEstimator* cpe) {
    return RecHitPointer( new TSiStripRecHit2DLocalPos( geom, rh, cpe));
  }

  static RecHitPointer build( const LocalPoint& pos, const LocalError& err,
			      const GeomDet* det,
			      const edm::Ref< edm::DetSetVector<SiStripCluster>,SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster>  >  clust,
			      const StripClusterParameterEstimator* cpe) {
    return RecHitPointer( new TSiStripRecHit2DLocalPos( pos, err, det, clust, cpe));
  }


private:

  SiStripRecHit2D*              theHitData;
  const StripClusterParameterEstimator* theCPE;

  TSiStripRecHit2DLocalPos (const GeomDet * geom, const SiStripRecHit2D* rh,
			    const StripClusterParameterEstimator* cpe) : 
    TransientTrackingRecHit(geom), theHitData(rh->clone()), theCPE(cpe) 
  {}

  /// Creates the TrackingRecHit internally, avoids redundent cloning
  TSiStripRecHit2DLocalPos( const LocalPoint& pos, const LocalError& err,
			    const GeomDet* det,
			    const edm::Ref< edm::DetSetVector<SiStripCluster>,SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster>  >  clust,
			    const StripClusterParameterEstimator* cpe);

  TSiStripRecHit2DLocalPos( const TSiStripRecHit2DLocalPos& other ) :
    TransientTrackingRecHit( other.det()), 
    theHitData( other.specificHit()->clone()),
    theCPE( other.cpe()) {}

  virtual TSiStripRecHit2DLocalPos* clone() const {
    return new TSiStripRecHit2DLocalPos(*this);
  }

};



#endif
