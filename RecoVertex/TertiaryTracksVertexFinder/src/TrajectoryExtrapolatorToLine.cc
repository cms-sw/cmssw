
#include "RecoVertex/TertiaryTracksVertexFinder/interface/TrajectoryExtrapolatorToLine.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "Geometry/Surface/interface/PlaneBuilder.h"
#include "Geometry/Surface/interface/Line.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "Geometry/CommonDetAlgo/interface/DeepCopyPointerByClone.h"

// need for propagator
#include "MagneticField/Engine/interface/MagneticField.h"

//#include "Utilities/Notification/interface/Verbose.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

using namespace reco;
using namespace std;

TrajectoryStateOnSurface 
TrajectoryExtrapolatorToLine::extrapolate(const FreeTrajectoryState& fts, 
					  const Line & L,
					  const MagneticField* field) const
{
  // use Bidirectional by default
  AnalyticalPropagator p(field,anyDirection);
  return this->extrapolate(fts, L, p);
}

TrajectoryStateOnSurface TrajectoryExtrapolatorToLine::extrapolate(
  const FreeTrajectoryState& fts, const Line & L, 
  const Propagator& aPropagator) const
{
  // construct BidirectionalPropagator
  DeepCopyPointerByClone<Propagator> p(aPropagator.clone());
  p->setPropagationDirection(anyDirection);

  // create a fts without errors for faster propagation
  FreeTrajectoryState fastFts(fts.parameters());
  GlobalVector T1 = fastFts.momentum().unit();
  GlobalPoint T0 = fastFts.position();
  double distance = 9999999.9;
  double old_distance;
  int n_iter = 0;
  bool refining = true;
  PlaneBuilder pBuilder;

  while (refining) {

    // describe orientation of target surface on basis of track parameters
    n_iter++;
    Line T(T0,T1);
    GlobalPoint B = T.closerPointToLine(L);
    old_distance = distance;

    //create surface
    GlobalPoint BB = B + 0.3 * (T0-B);
    Surface::PositionType pos(BB);
    GlobalVector XX(T1.y(),-T1.x(),0.);
    GlobalVector YY(T1.cross(XX));
    Surface::RotationType rot(XX,YY);
    ReferenceCountingPointer<BoundPlane> surface = pBuilder.plane(pos, rot);

    // extrapolate fastFts to target surface
    TrajectoryStateOnSurface tsos = p->propagate(fastFts, *surface);

    if (!tsos.isValid()) {
      //if ( testV )  cout << "TETL - extrapolation failed" << endl;
      return tsos;
    } else {
      T0 = tsos.globalPosition();
      T1 = tsos.globalMomentum().unit();
      GlobalVector D = L.distance(T0);
      distance = D.mag();
      if (abs(old_distance - distance) < 0.000001) {refining = false;}
      if (old_distance-distance<0.){refining=false;
	//if ( testV )  cout<< "TETL- stop to skip loops"<<endl;
//      return tsos;
	}
    }
  }
  //
  // Now propagate with errors and (not for the moment) perform rotation
  //
  // Origin of local system: point of closest approach on the line
  // (w.r.t. to tangent to helix at last iteration)
  //
  Line T(T0,T1);
  GlobalPoint origin(L.closerPointToLine(T));
  //
  // Axes of local system: 
  //   x from line to helix at closest approach
  //   z along the helix
  //   y to complete right-handed system
  //
  GlobalVector ZZ(T1.unit());
  GlobalVector YY(ZZ.cross(T0-origin).unit());
  GlobalVector XX(YY.cross(ZZ));
  Surface::RotationType rot(XX,YY,ZZ);
  ReferenceCountingPointer<BoundPlane> surface = pBuilder.plane(origin,rot);
  TrajectoryStateOnSurface tsos = p->propagate(fts, *surface);

  return tsos;
}

//---------------------------------------------------------------------------------------------

TrajectoryStateOnSurface TrajectoryExtrapolatorToLine::stateAtLine(const TransientTrack& theTrack, const Line & aLine) const 
{
  const MagneticField* field = theTrack.field();

  // approach from impact point first
  // create a fts without errors for faster propagation
  FreeTrajectoryState fastFts(theTrack.impactPointState().freeTrajectoryState()->parameters());
  //TrajectoryExtrapolatorToLine tetl;
  TrajectoryStateOnSurface cptl = this->extrapolate(fastFts, aLine,field);

  // extrapolate from closest measurement
  if (cptl.isValid()) {

    // FreeTrajectoryState* fts = theTrack.closestState(cptl.globalPosition()).freeState();
    // closestState() not provided in CMSSW, need to do by hand ...
    FreeTrajectoryState* fts;
    GlobalVector d1 = theTrack.innermostMeasurementState().globalPosition()-cptl.globalPosition();
    GlobalVector d2 = theTrack.outermostMeasurementState().globalPosition()-cptl.globalPosition();
    if (d1.mag() < d2.mag()) fts=theTrack.innermostMeasurementState().freeState();
    else                     fts=theTrack.outermostMeasurementState().freeState();

    TrajectoryStateOnSurface cptl2 = this->extrapolate(*fts, aLine,field);
    return cptl2;
  } else { 
    return cptl;
  } 
}

 
