#ifndef TSiTrackerMultiRecHit_h
#define TSiTrackerMultiRecHit_h

#include "TrackingTools/TransientTrackingRecHit/interface/TValidTrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/HelpertRecHit2DLocalPos.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

/*
A TransientTrackingRecHit for the SiTrackerMultiRecHit
*/

class TSiTrackerMultiRecHit GCC11_FINAL : public TValidTrackingRecHit {
public:
  //virtual ~TSiTrackerMultiRecHit() {delete theHitData;}
  virtual ~TSiTrackerMultiRecHit() {}
  
  virtual AlgebraicVector parameters() const {return theHitData.parameters();}
  virtual AlgebraicSymMatrix parametersError() const {
    return HelpertRecHit2DLocalPos().parError( theHitData.localPositionError(), *det());
    //return theHitData.parametersError();
  }
  
  virtual void getKfComponents( KfComponentsHolder & holder ) const {
    HelpertRecHit2DLocalPos().getKfComponents(holder, theHitData, *det()); 
  }
  virtual DetId geographicalId() const {return theHitData.geographicalId();}
  virtual AlgebraicMatrix projectionMatrix() const {return theHitData.projectionMatrix();}
  virtual int dimension() const {return theHitData.dimension();}
  
  virtual LocalPoint localPosition() const {return theHitData.localPosition();}
  virtual LocalError localPositionError() const {return theHitData.localPositionError();}
  
  virtual const TrackingRecHit * hit() const {return &theHitData;};
  const SiTrackerMultiRecHit* specificHit() const {return &theHitData;}
  
  virtual bool isValid() const{return theHitData.isValid();}
  
  virtual std::vector<const TrackingRecHit*> recHits() const {
    return theHitData.recHits();
  } 
  virtual std::vector<TrackingRecHit*> recHits() {
    return theHitData.recHits();
  }	
  
  
  /// interface needed to set and read back an annealing value that has been applied to the current hit error matrix when
  /// using it as a component for a composite rec hit (useful for the DAF)
  void setAnnealingFactor(float annealing) {annealing_ = annealing;} 
  float getAnnealingFactor() const {return annealing_;} 
  

  //vector of weights
  std::vector<float> const & weights() const {return theHitData.weights();}
  std::vector<float>  & weights() {return theHitData.weights();}

  //returns the weight for the i component
  float  weight(unsigned int i) const {return theHitData.weight(i);}
  float  & weight(unsigned int i) {return theHitData.weight(i);}

  
  virtual const GeomDetUnit* detUnit() const;
  
  virtual bool canImproveWithTrack() const {return true;}
  
  virtual RecHitPointer clone(const TrajectoryStateOnSurface& ts) const;
  
  virtual ConstRecHitContainer transientHits() const {return theComponents;};
  
  static RecHitPointer build( const GeomDet * geom, const SiTrackerMultiRecHit* rh, 
			      const ConstRecHitContainer& components, float annealing=1.){
    return RecHitPointer(new TSiTrackerMultiRecHit( geom, rh, components, annealing));
  }
  
  
  
private:
  SiTrackerMultiRecHit theHitData;
  //holds the TransientTrackingRecHit components of the MultiRecHit 
  ConstRecHitContainer theComponents;   
  float annealing_;
 
  TSiTrackerMultiRecHit(const GeomDet * geom, const SiTrackerMultiRecHit* rh,  
			const ConstRecHitContainer& components, float annealing):
    TValidTrackingRecHit(*geom), theHitData(*rh), theComponents(components), annealing_(annealing){}
  
  virtual TSiTrackerMultiRecHit* clone() const {
    return new TSiTrackerMultiRecHit(*this);
  }
  
  
};		

#endif
