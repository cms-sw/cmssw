#ifndef DetLayers_GeometricSearchDet_h
#define DetLayers_GeometricSearchDet_h

#include "Geometry/Surface/interface/BoundSurface.h"
#include "TrackingTools/DetLayers/interface/DetGroup.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include <vector>

using namespace std;
class GSDUnit;
class MeasurementEstimator;
class Propagator;
class DetGroup;

class GeometricSearchDet {

 public:
  
  typedef BoundSurface::PositionType        PositionType;
  typedef BoundSurface::RotationType        RotationType;
  typedef TrajectoryStateOnSurface          TrajectoryState;
  
  virtual ~GeometricSearchDet() {};
  
  /// The surface of the GeometricSearchDet
  virtual const BoundSurface& surface() const = 0;
  
  /// Returns position of the surface
  virtual const Surface::PositionType& position() const {return surface().position();}

  /// Returns basic components, if any
  virtual std::vector< const GSDUnit*> basicComponents() const = 0;


  /** tests the geometrical compatibility of the Det with the predicted state.
   *  The  FreeTrajectoryState argument is propagated to the Det surface using
   *  the Propagator argument. The resulting TrajectoryStateOnSurface is
   *  tested for compatibility with the surface bounds.
   *  If compatible, a pair< true, propagatedState> is returned.
   *  If the propagation fails, or if the state is not compatible,
   *  a pair< false, propagatedState> is returned.
   */
  virtual pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const=0;

  virtual bool hasGroups() const = 0;




};

#endif
