
#include "TrackingTools/PatternTools/interface/TrajectoryExtrapolatorToLine.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/OpenBounds.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

TrajectoryStateOnSurface TrajectoryExtrapolatorToLine::extrapolate(const FreeTrajectoryState& fts,
								   const Line & L, 
								   const Propagator& aPropagator) const
{
  DeepCopyPointerByClone<Propagator> p(aPropagator.clone());
  p->setPropagationDirection(anyDirection);

  FreeTrajectoryState fastFts(fts.parameters(), fts.curvilinearError());
  GlobalVector T1 = fastFts.momentum().unit();
   GlobalPoint T0 = fastFts.position();
   double distance = 9999999.9;
   double old_distance;
   int n_iter = 0;
   bool refining = true;

   LogDebug("TrajectoryExtrapolatorToLine") << "START REFINING";  
   while (refining) {
     LogDebug("TrajectoryExtrapolatorToLine") << "Refining cycle...";
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
     ReferenceCountingPointer<Plane> surface = Plane::build(pos, rot);
     LogDebug("TrajectoryExtrapolatorToLine") << "Current plane position: " << surface->toGlobal(LocalPoint(0.,0.,0.));
     LogDebug("TrajectoryExtrapolatorToLine") << "Current plane normal: " <<  surface->toGlobal(LocalVector(0,0,1));
     LogDebug("TrajectoryExtrapolatorToLine") << "Current momentum:     " <<  T1;



     // extrapolate fastFts to target surface
     TrajectoryStateOnSurface tsos = p->propagate(fastFts, *surface);

     if (!tsos.isValid()) {
       LogDebug("TrajectoryExtrapolatorToLine") << "TETL - extrapolation failed";
       return tsos;
     } else {
       T0 = tsos.globalPosition();
       T1 = tsos.globalMomentum().unit();
       GlobalVector D = L.distance(T0);
       distance = D.mag();
       if (fabs(old_distance - distance) < 0.000001) {refining = false;}
       if (old_distance-distance<0.){
	 refining=false;
	 LogDebug("TrajectoryExtrapolatorToLine")<< "TETL- stop to skip loops";
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
   ReferenceCountingPointer<Plane> surface = Plane::build(origin, rot);
   TrajectoryStateOnSurface tsos = p->propagate(fts, *surface);
 
   return tsos;

}
