#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripRecHit2DLocalPos_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripRecHit2DLocalPos_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TValidTrackingRecHit.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/HelpertRecHit2DLocalPos.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class TSiStripRecHit2DLocalPos GCC11_FINAL : public TValidTrackingRecHit {
public:
  
  typedef SiStripRecHit2D::ClusterRef SiStripClusterRef;
  
  virtual ~TSiStripRecHit2DLocalPos() {}
  
  
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
  
  virtual const GeomDetUnit* detUnit() const { return static_cast<const GeomDetUnit*>(det()); }
  
  virtual bool canImproveWithTrack() const {return true;}
  
  //RC virtual TSiStripRecHit2DLocalPos* clone(const TrajectoryStateOnSurface& ts) const;
  virtual RecHitPointer clone(const TrajectoryStateOnSurface& ts) const;
  
  // Extension of the TransientTrackingRecHit interface
  
  const SiStripRecHit2D* specificHit() const {return &theHitData;};
  const StripClusterParameterEstimator* cpe() const {return theCPE;}
  
  static RecHitPointer build( const GeomDet * geom, const SiStripRecHit2D* rh,
			      const StripClusterParameterEstimator* cpe,
			      bool computeCoarseLocalPosition=false) {
    return RecHitPointer( new TSiStripRecHit2DLocalPos( geom, rh, cpe,computeCoarseLocalPosition));
  }
  
  
  static RecHitPointer build( const LocalPoint& pos, const LocalError& err,
			      const GeomDet* det,
			      const OmniClusterRef & clust,
			      const StripClusterParameterEstimator* cpe) {
    return RecHitPointer( new TSiStripRecHit2DLocalPos( pos, err, det, clust, cpe));
  }

  static RecHitPointer build( const LocalPoint& pos, const LocalError& err,
			      const GeomDet* det,
			      const SiStripClusterRef & clust,
			      const StripClusterParameterEstimator* cpe) {
    return RecHitPointer( new TSiStripRecHit2DLocalPos( pos, err, det, OmniClusterRef(clust), cpe));
  }
  
  
  static float sigmaPitch(LocalPoint const& pos, LocalError err, 
			  GeomDetUnit const & stripdet) {

    const StripTopology& topol=(const StripTopology&)stripdet.topology();
    
    HelpertRecHit2DLocalPos::updateWithAPE(err,stripdet);
    MeasurementError error=topol.measurementError(pos,err);
    auto pitch=topol.localPitch(pos);
    return error.uu()*pitch*pitch;
  }

  
private:
  
  const StripClusterParameterEstimator* theCPE;
  SiStripRecHit2D              theHitData;
 


 
  TSiStripRecHit2DLocalPos (const GeomDet * geom, const SiStripRecHit2D* rh,
			    const StripClusterParameterEstimator* cpe,
			    bool computeCoarseLocalPosition) : 
    TValidTrackingRecHit(geom), theCPE(cpe) 
  {
    if ( (rh->hasPositionAndError()) | (!computeCoarseLocalPosition) ) {
      theHitData = SiStripRecHit2D(*rh);
      return;
    }

    if unlikely(computeCoarseLocalPosition & (!cpe) ){
      edm::LogError("TSiStripRecHit2DLocalPos")<<" trying to compute coarse local position but CPE is not provided. Not computing local position from disk for the transient tracking rechit.";
      theHitData = SiStripRecHit2D(*rh);
      return;
    }
    
    const GeomDetUnit* gdu = static_cast<const GeomDetUnit*>(geom);
    LogDebug("TSiStripRecHit2DLocalPos")<<"calculating coarse position/error.";

    StripClusterParameterEstimator::LocalValues lval= theCPE->localParameters(rh->stripCluster(), *gdu);
    theHitData = SiStripRecHit2D(lval.first, lval.second, *geom, rh->omniCluster());

  }
  
  /// Creates the TrackingRecHit internally, avoids redundent cloning
  TSiStripRecHit2DLocalPos( const LocalPoint& pos, const LocalError& err,
			    const GeomDet* idet,
			    const OmniClusterRef & clust,
			    const StripClusterParameterEstimator* cpe) :
    TValidTrackingRecHit(idet), 
    theCPE(cpe), theHitData(pos, err, *idet, clust) {} 
    
  virtual TSiStripRecHit2DLocalPos* clone() const {
    return new TSiStripRecHit2DLocalPos(*this);
  }
  
  virtual ConstRecHitContainer transientHits() const;
  
};

#endif
