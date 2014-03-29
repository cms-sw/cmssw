#ifndef GSTransientTrackingRecHit_H
#define GSTransientTrackingRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/TValidTrackingRecHit.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h" 

// indentical to GenericTransientTrackingRecHit but for getKf
class GSTransientTrackingRecHit final : public TValidTrackingRecHit{
public:
  typedef TrackingRecHit::Type Type;

  virtual ~GSTransientTrackingRecHit() {delete trackingRecHit_;}

  virtual AlgebraicVector parameters() const {return trackingRecHit_->parameters();}
  virtual AlgebraicSymMatrix parametersError() const {return trackingRecHit_->parametersError();}
  virtual AlgebraicMatrix projectionMatrix() const {return trackingRecHit_->projectionMatrix();}
  virtual int dimension() const {return trackingRecHit_->dimension();}

  void getKfComponents( KfComponentsHolder & holder ) const { trackingRecHit_->getKfComponents(holder); }

  virtual LocalPoint localPosition() const {return trackingRecHit_->localPosition();}
  virtual LocalError localPositionError() const {return trackingRecHit_->localPositionError();}

  virtual bool canImproveWithTrack() const {return false;}

  virtual const TrackingRecHit * hit() const {return trackingRecHit_;}
  TrackingRecHit * cloneHit() const { return hit()->clone();}

  virtual std::vector<const TrackingRecHit*> recHits() const {
    return ((const TrackingRecHit *)(trackingRecHit_))->recHits();
  }
  virtual std::vector<TrackingRecHit*> recHits() {
    return trackingRecHit_->recHits();
  }

  static RecHitPointer build( const GeomDet * geom, const TrackingRecHit * rh) {
    return RecHitPointer( new GSTransientTrackingRecHit( geom, *rh));
  }

protected:

  // private constructors enforce usage of builders
  GSTransientTrackingRecHit(const GeomDet * geom, const TrackingRecHit& rh) :
    TValidTrackingRecHit(geom,rh) {
    trackingRecHit_ = rh.clone();
  }
  
  /// for derived classes convenience, does not clone!
  GSTransientTrackingRecHit(const GeomDet * geom, TrackingRecHit* rh) :
    TValidTrackingRecHit(geom,*rh), trackingRecHit_(rh) {}
  
  GSTransientTrackingRecHit( const GenericTransientTrackingRecHit & other ) :
  TValidTrackingRecHit( other.det(),other) {
    trackingRecHit_ = other.cloneHit();
  }
  
  TrackingRecHit * trackingRecHit_;
  
 private:
  
  // should not have assignment operator (?)
  GSTransientTrackingRecHit & operator= (const GenericTransientTrackingRecHit & t) {
    trackingRecHit_ = t.cloneHit();
    return *(this);
  }

  // hide the clone method for ReferenceCounted. Warning: this method is still 
  // accessible via the bas class TrackingRecHit interface!
   virtual GSTransientTrackingRecHit * clone() const {
     return new GSTransientTrackingRecHit(*this);
   }

};

#endif

