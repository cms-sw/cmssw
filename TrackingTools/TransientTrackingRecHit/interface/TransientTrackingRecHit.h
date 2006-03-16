#ifndef TransientTrackingRecHit_H
#define TransientTrackingRecHit_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include <Geometry/Vector/interface/GlobalPoint.h>
#include <TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h>
#include <Geometry/CommonDetAlgo/interface/AlgebraicObjects.h>
#include "DataFormats/Common/interface/OwnVector.h"

class TransientTrackingRecHit : public TrackingRecHit
{
 public:
  TransientTrackingRecHit(edm::ESHandle<TrackingGeometry> geom, TrackingRecHit * rh) {
    _geom = geom ;
    _trackingRecHit = rh->clone();
  }
  virtual ~TransientTrackingRecHit() {delete _trackingRecHit;}

//   virtual const GlobalPoint globalPoint() = 0;

/*  void setDet() { */
/*     _id = geographicalId(); */
/*     _detMap[_id] = _geom->idToDet(_id); */
/*   }; */

  const GeomDetUnit * detUnit() const;
  const GeomDet * det() const;

  virtual AlgebraicVector parameters() const {return _trackingRecHit->parameters();}
  virtual AlgebraicSymMatrix parametersError() const {return _trackingRecHit->parametersError();}
  virtual DetId geographicalId() const {return _trackingRecHit->geographicalId();}
  virtual AlgebraicMatrix projectionMatrix() const {return _trackingRecHit->projectionMatrix();}
  virtual int dimension() const {return _trackingRecHit->dimension();}
  virtual std::vector<const TrackingRecHit*> recHits() const {return ((const TrackingRecHit *)(_trackingRecHit))->recHits();}
  virtual std::vector<TrackingRecHit*> recHits() {return _trackingRecHit->recHits();}
  virtual LocalPoint localPosition() const {return _trackingRecHit->localPosition();}
  virtual LocalError localPositionError() const {return _trackingRecHit->localPositionError();}


  virtual GlobalPoint globalPosition() const ;

  virtual GlobalError globalPositionError() const ;


  virtual AlgebraicVector parameters(const TrajectoryStateOnSurface& ts) const  = 0;
  virtual AlgebraicSymMatrix parametersError(const TrajectoryStateOnSurface& ts) const = 0;
  TrackingRecHit * hit() const {return _trackingRecHit;};
  
  bool isValid() {return true;}
  bool isValid() const{return true;}

  virtual TransientTrackingRecHit * clone() const = 0;
  virtual edm::OwnVector<const TransientTrackingRecHit> transientHits() const {
    edm::OwnVector<const TransientTrackingRecHit> temp;
    temp.push_back(this);
    return temp;
  }
  virtual edm::OwnVector<TransientTrackingRecHit> transientHits() {
    edm::OwnVector<TransientTrackingRecHit> temp;
    temp.push_back(this);
    return temp;
  }

  private:
  edm::ESHandle<TrackingGeometry> _geom ;
  TrackingRecHit * _trackingRecHit;
};

#endif

