#ifndef RKPropagatorInS_H
#define RKPropagatorInS_H

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/TrajectoryState/interface/TrackCharge.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "FWCore/Utilities/interface/Visibility.h"


class GlobalTrajectoryParameters;
class GlobalParametersWithPath;
class MagVolume;
class RKLocalFieldProvider;
class CartesianStateAdaptor;

class RKPropagatorInS GCC11_FINAL : public Propagator {
public:

  // RKPropagatorInS( PropagationDirection dir = alongMomentum) : Propagator(dir), theVolume(0) {}
  // tolerance (see below) used to be 1.e-5 --> this was observed to cause problems with convergence 
  // when propagating to cylinder with large radius (~10 meter) MM 22/6/07

  explicit RKPropagatorInS( const MagVolume& vol, PropagationDirection dir = alongMomentum,
			    double tolerance = 5.e-5) : 
    Propagator(dir), theVolume( &vol), theTolerance( tolerance) {}

  ~RKPropagatorInS() {}

  using Propagator::propagate;
  using Propagator::propagateWithPath;

  virtual TrajectoryStateOnSurface 
  propagate (const FreeTrajectoryState&, const Plane&) const;

  virtual TrajectoryStateOnSurface 
  propagate (const FreeTrajectoryState&, const Cylinder&) const;

  virtual std::pair< TrajectoryStateOnSurface, double> 
  propagateWithPath (const FreeTrajectoryState&, const Plane&) const;

  virtual std::pair< TrajectoryStateOnSurface, double> 
  propagateWithPath (const FreeTrajectoryState&, const Cylinder&) const;

  TrajectoryStateOnSurface propagate(const TrajectoryStateOnSurface& ts, 
                                     const Plane& plane) const {
    return propagateWithPath( *ts.freeState(),plane).first;
  }
  


  virtual Propagator * clone() const;

  virtual const MagneticField* magneticField() const {return theVolume;}

private:

  typedef std::pair<TrajectoryStateOnSurface,double>     TsosWP;

  const MagVolume* theVolume;
  double           theTolerance;

  GlobalTrajectoryParameters gtpFromLocal( const Basic3DVector<double>& lpos,
					   const Basic3DVector<double>& lmom,
					   TrackCharge ch, const Surface& surf) const dso_internal;

  GlobalTrajectoryParameters gtpFromVolumeLocal( const CartesianStateAdaptor& state, 
						 TrackCharge charge) const  dso_internal;
    
  RKLocalFieldProvider fieldProvider() const;
  RKLocalFieldProvider fieldProvider( const Cylinder& cyl) const dso_internal;

  PropagationDirection invertDirection( PropagationDirection dir) const dso_internal;

  Basic3DVector<double> rkPosition( const GlobalPoint& pos) const dso_internal;
  Basic3DVector<double> rkMomentum( const GlobalVector& mom) const dso_internal;
  GlobalPoint           globalPosition( const Basic3DVector<double>& pos) const dso_internal;
  GlobalVector          globalMomentum( const Basic3DVector<double>& mom) const dso_internal;

  GlobalParametersWithPath propagateParametersOnPlane( const FreeTrajectoryState& ts, 
						       const Plane& plane) const dso_internal;
  GlobalParametersWithPath propagateParametersOnCylinder( const FreeTrajectoryState& ts, 
							  const Cylinder& cyl) const dso_internal;

};

#endif
