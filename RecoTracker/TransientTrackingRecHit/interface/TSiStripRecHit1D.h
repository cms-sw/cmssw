#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripRecHit1D_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripRecHit1D_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TValidTrackingRecHit.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/HelpertRecHit2DLocalPos.h"

class TSiStripRecHit1D GCC11_FINAL : public TValidTrackingRecHit {
public:

  typedef SiStripRecHit1D::ClusterRef SiStripClusterRef;
  
  virtual ~TSiStripRecHit1D() {}

  
  virtual void getKfComponents( KfComponentsHolder & holder ) const {
      HelpertRecHit2DLocalPos().getKfComponents(holder, theHitData, *det()); 
  }
 

  virtual AlgebraicVector parameters() const {return theHitData.parameters();}

  
  virtual AlgebraicSymMatrix parametersError() const {
    return HelpertRecHit2DLocalPos().parError( theHitData.localPositionError(), *det()); 
    //    return theHitData->parametersError();
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

  virtual bool canImproveWithTrack() const {return true;}

  //RC virtual TSiStripRecHit2DLocalPos* clone(const TrajectoryStateOnSurface& ts) const;
  virtual RecHitPointer clone(const TrajectoryStateOnSurface& ts) const;

  // Extension of the TransientTrackingRecHit interface

  const SiStripRecHit1D* specificHit() const {return &theHitData;};
  const StripClusterParameterEstimator* cpe() const {return theCPE;}

  static RecHitPointer build( const GeomDet * geom, const SiStripRecHit1D* rh,
			      const StripClusterParameterEstimator* cpe,
			      bool computeCoarseLocalPosition=false) {
    return RecHitPointer( new TSiStripRecHit1D( geom, rh, cpe,computeCoarseLocalPosition));
  }

  static RecHitPointer build( const LocalPoint& pos, const LocalError& err,
			      const GeomDet* det,
			      const OmniClusterRef & clust,
			      const StripClusterParameterEstimator* cpe) {
    return RecHitPointer( new TSiStripRecHit1D( pos, err, det, clust, cpe));
  }

  static RecHitPointer build( const LocalPoint& pos, const LocalError& err,
			      const GeomDet* det,
			      const SiStripClusterRef & clust,
			      const StripClusterParameterEstimator* cpe) {
    return RecHitPointer( new TSiStripRecHit1D( pos, err, det, OmniClusterRef(clust), cpe));
  }


private:

  const StripClusterParameterEstimator* theCPE;
  SiStripRecHit1D              theHitData;

  TSiStripRecHit1D (const GeomDet * geom, const SiStripRecHit1D* rh,
		    const StripClusterParameterEstimator* cpe,
		    bool computeCoarseLocalPosition);

  /// Creates the TrackingRecHit internally, avoids redundent cloning
  TSiStripRecHit1D( const LocalPoint& pos, const LocalError& err,
		    const GeomDet* det,
		    const OmniClusterRef & clust,
		    const StripClusterParameterEstimator* cpe) :
    TValidTrackingRecHit(det), 
    theCPE(cpe), theHitData(pos, err, *det, clust){} 


  

  virtual TSiStripRecHit1D* clone() const {
    return new TSiStripRecHit1D(*this);
  }

};

#endif
