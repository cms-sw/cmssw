#ifndef TransientTrackingRecHitByValue_H
#define TransientTrackingRecHitByValue_H

#include "TrackingTools/TransientTrackingRecHit/interface/TValidTrackingRecHit.h"

template<typename RecHit>
class TransientTrackingRecHitByValue: public TValidTrackingRecHit{
public:
  typedef TrackingRecHit::Type Type;
  
  virtual ~TransientTrackingRecHitByValue() {}
  
  virtual AlgebraicVector parameters() const GCC11_FINAL {return m_trackingRecHit.parameters();}
  virtual AlgebraicSymMatrix parametersError() const GCC11_FINAL {return m_trackingRecHit.parametersError();}
  virtual AlgebraicMatrix projectionMatrix() const GCC11_FINAL {return m_trackingRecHit.projectionMatrix();}
  virtual int dimension() const GCC11_FINAL {return m_trackingRecHit.dimension();}
  
  virtual LocalPoint localPosition() const GCC11_FINAL {return m_trackingRecHit.localPosition();}
  virtual LocalError localPositionError() const GCC11_FINAL {return m_trackingRecHit.localPositionError();}
  
  virtual bool canImproveWithTrack() const {return false;}
  
  virtual const RecHit * hit() const GCC11_FINAL {return &m_trackingRecHit;};
  RecHit * cloneHit() const  GCC11_FINAL { return m_trackingRecHit.clone();}
  
  virtual std::vector<const TrackingRecHit*> recHits() const {
    return hit()->recHits();
  }
  virtual std::vector<TrackingRecHit*> recHits() {
    return m_trackingRecHit.recHits();
  }
  
  static RecHitPointer build( const GeomDet * geom, const RecHit * rh) {
    return RecHitPointer( new TransientTrackingRecHitByValue<RecHit>( geom, *rh));
  }
  
protected:
  
  // private constructors enforce usage of builders
  TransientTrackingRecHitByValue(const GeomDet * geom, const RecHit& rh) :
    TValidTrackingRecHit(geom,rh), m_trackingRecHit(rh) {
  }
  
  
  TransientTrackingRecHitByValue( const TransientTrackingRecHitByValue<RecHit> & other ) :
    TValidTrackingRecHit( other.det(),other), m_trackingRecHit(*other.hit()) {
  }
  
private:
  
  RecHit m_trackingRecHit;
  
  // should not have assignment operator (?)
  TransientTrackingRecHitByValue<RecHit> & operator= (const TransientTrackingRecHitByValue<RecHit> & t) {
    m_trackingRecHit = *t.hit();
    return *(this);
  }
  
  // hide the clone method for ReferenceCounted. Warning: this method is still 
  // accessible via the bas class TrackingRecHit interface!
  virtual TransientTrackingRecHitByValue<RecHit> * clone() const {
    return new TransientTrackingRecHitByValue<RecHit>(*this);
  }
  
};

#endif // TransientTrackingRecHitByValue_H

