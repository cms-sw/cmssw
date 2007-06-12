#ifndef RKPropagatorInS_H
#define RKPropagatorInS_H

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/TrajectoryState/interface/TrackCharge.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"


class GlobalTrajectoryParameters;
class GlobalParametersWithPath;
class MagVolume;
class RKLocalFieldProvider;
class CartesianStateAdaptor;

class RKPropagatorInS : public Propagator {
public:

  // RKPropagatorInS( PropagationDirection dir = alongMomentum) : Propagator(dir), theVolume(0) {}

  explicit RKPropagatorInS( const MagVolume& vol, PropagationDirection dir = alongMomentum,
			    double tolerance = 1.e-5) : 
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
					   TrackCharge ch, const Surface& surf) const;
  GlobalTrajectoryParameters gtpFromVolumeLocal( const CartesianStateAdaptor& state, 
						 TrackCharge charge) const;
    
  RKLocalFieldProvider fieldProvider() const;
  RKLocalFieldProvider fieldProvider( const Cylinder& cyl) const;

  PropagationDirection invertDirection( PropagationDirection dir) const;

  Basic3DVector<double> rkPosition( const GlobalPoint& pos) const;
  Basic3DVector<double> rkMomentum( const GlobalVector& mom) const;
  GlobalPoint           globalPosition( const Basic3DVector<double>& pos) const;
  GlobalVector          globalMomentum( const Basic3DVector<double>& mom) const;

  GlobalParametersWithPath propagateParametersOnPlane( const FreeTrajectoryState& ts, 
						       const Plane& plane) const;
  GlobalParametersWithPath propagateParametersOnCylinder( const FreeTrajectoryState& ts, 
							  const Cylinder& cyl) const;

};

#endif
