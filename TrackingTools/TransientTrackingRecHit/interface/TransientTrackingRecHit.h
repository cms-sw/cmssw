#ifndef TransientTrackingRecHit_H
#define TransientTrackingRecHit_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include <Geometry/Vector/interface/GlobalPoint.h>
#include <TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h>
#include <Geometry/CommonDetAlgo/interface/AlgebraicObjects.h>

class TransientTrackingRecHit : public TrackingRecHit
{
 public:
  TransientTrackingRecHit(edm::ESHandle<TrackingGeometry> geom, TrackingRecHit * rh) {
    _geom = geom ;
    _trackingRecHit = rh;
  }
  virtual ~TransientTrackingRecHit() {;}

//   virtual const GlobalPoint globalPoint() = 0;

/*  void setDet() { */
/*     _id = geographicalId(); */
/*     _detMap[_id] = _geom->idToDet(_id); */
/*   }; */

  const GeomDetUnit * detUnit() const;

  virtual AlgebraicVector parameters() const {return _trackingRecHit->parameters();}
  virtual AlgebraicSymMatrix parametersError() const {return _trackingRecHit->parametersError();}
  virtual DetId geographicalId() const {return _trackingRecHit->geographicalId();}
  virtual TrackingRecHit * clone() const {return _trackingRecHit->clone();}
  virtual AlgebraicMatrix projectionMatrix() const {return _trackingRecHit->projectionMatrix();}
  virtual int dimension() const {return _trackingRecHit->dimension();}
  virtual std::vector<const TrackingRecHit*> recHits() const {return ((const TrackingRecHit *)(_trackingRecHit))->recHits();}
  virtual std::vector<TrackingRecHit*> recHits() {return _trackingRecHit->recHits();}
  virtual LocalPoint localPosition() const {return _trackingRecHit->localPosition();}
  virtual LocalError localPositionError() const {return _trackingRecHit->localPositionError();}

  virtual AlgebraicVector parameters(const TrajectoryStateOnSurface& ts) const = 0;
  virtual AlgebraicSymMatrix parametersError(const TrajectoryStateOnSurface& ts) const = 0;
  TrackingRecHit * hit() {return _trackingRecHit;};

/*   virtual LocalPoint localPosition() = 0; */

  private:
/*   DetId _id ; */
/*   std::map<DetId ,const  GeomDetUnit *> _detMap ; */
  edm::ESHandle<TrackingGeometry> _geom ;
  TrackingRecHit * _trackingRecHit;
};

#endif

