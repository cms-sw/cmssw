#ifndef TrackKinematicStatePropagator_H
#define TrackKinematicStatePropagator_H

#include "RecoVertex/KinematicFitPrimitives/interface/KinematicStatePropagator.h"
#include "DataFormats/GeometryVector/interface/GlobalTag.h"
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "DataFormats/GeometryVector/interface/Vector3DBase.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/GeomPropagators/interface/HelixBarrelPlaneCrossingByCircle.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

/**
 * Propagator for TransientTrack based KinematicStates.
 * Does not include the material.
 */


class  TrackKinematicStatePropagator final : public KinematicStatePropagator
{
public:

  TrackKinematicStatePropagator() {}
  
  ~TrackKinematicStatePropagator() override {}
  
  
  /**
   * Propagation to the point of closest approach in
   * transverse plane to the given point
   */   
  KinematicState propagateToTheTransversePCA(const KinematicState& state, const GlobalPoint& referencePoint) const override;
  
  bool willPropagateToTheTransversePCA(const KinematicState& state, const GlobalPoint& point) const override;
  
    
  /**
   * Clone method reimplemented from
   * abstract class
   */
  KinematicStatePropagator * clone() const override
  {return new TrackKinematicStatePropagator(*this);}
  
 private:
  
  /**
   * Internal private methods, distinguishing between the propagation of neutrals
   * and propagation of cahrged tracks.
   */
  virtual KinematicState propagateToTheTransversePCACharged(const KinematicState& state, const GlobalPoint& referencePoint) const;
  
  virtual KinematicState propagateToTheTransversePCANeutral(const KinematicState& state, const GlobalPoint& referencePoint) const;
  
  typedef Point3DBase< double, GlobalTag>    GlobalPointDouble;
  typedef Vector3DBase< double, GlobalTag>    GlobalVectorDouble;
 
};
#endif
