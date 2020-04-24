#ifndef GenericTransientTrackingRecHit_H
#define GenericTransientTrackingRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/TValidTrackingRecHit.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h" 

class GenericTransientTrackingRecHit: public TValidTrackingRecHit{
public:
  typedef TrackingRecHit::Type Type;

  ~GenericTransientTrackingRecHit() override {delete trackingRecHit_;}

  AlgebraicVector parameters() const override {return trackingRecHit_->parameters();}
  AlgebraicSymMatrix parametersError() const override {return trackingRecHit_->parametersError();}
  AlgebraicMatrix projectionMatrix() const override {return trackingRecHit_->projectionMatrix();}
  int dimension() const override {return trackingRecHit_->dimension();}

  // virtual void getKfComponents( KfComponentsHolder & holder ) const  override { trackingRecHit_->getKfComponents(holder); }
  // NO, because someone might specialize parametersError, projectionMatrix or parameters in the transient rechit
  // and in fact this happens for alignment

  LocalPoint localPosition() const  override {return trackingRecHit_->localPosition();}
  LocalError localPositionError() const  override {return trackingRecHit_->localPositionError();}

  bool canImproveWithTrack() const override {return false;}

  const TrackingRecHit * hit() const override {return trackingRecHit_;}
  TrackingRecHit * cloneHit() const override { return hit()->clone();}

  std::vector<const TrackingRecHit*> recHits() const override {
    return ((const TrackingRecHit *)(trackingRecHit_))->recHits();
  }
  std::vector<TrackingRecHit*> recHits() override {
    return trackingRecHit_->recHits();
  }

  static RecHitPointer build( const GeomDet * geom, const TrackingRecHit * rh) {
    return RecHitPointer( new GenericTransientTrackingRecHit( *geom, *rh));
  }

protected:

  // private constructors enforce usage of builders
  GenericTransientTrackingRecHit(const GeomDet & geom, const TrackingRecHit& rh) :
    TValidTrackingRecHit(geom,rh) {
    trackingRecHit_ = rh.clone();
  }
  
  /// for derived classes convenience, does not clone!
  GenericTransientTrackingRecHit(const GeomDet & geom, TrackingRecHit* rh) :
    TValidTrackingRecHit(geom,*rh), trackingRecHit_(rh) {}
  
  GenericTransientTrackingRecHit( const GenericTransientTrackingRecHit & other ) :
  TValidTrackingRecHit( *other.det(),other) {
    trackingRecHit_ = other.cloneHit();
  }
  
  TrackingRecHit * trackingRecHit_;
  
 private:
  
  // should not have assignment operator (?)
  GenericTransientTrackingRecHit & operator= (const GenericTransientTrackingRecHit & t) {
    trackingRecHit_ = t.cloneHit();
    return *(this);
  }

  // hide the clone method for ReferenceCounted. Warning: this method is still 
  // accessible via the bas class TrackingRecHit interface!
   GenericTransientTrackingRecHit * clone() const override {
     return new GenericTransientTrackingRecHit(*this);
   }

};

#endif

