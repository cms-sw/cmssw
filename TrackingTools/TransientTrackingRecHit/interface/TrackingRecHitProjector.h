#ifndef TrackingRecHitProjector_H
#define TrackingRecHitProjector_H

#include "TrackingTools/TransientTrackingRecHit/interface/TValidTrackingRecHit.h"
//#include <iostream>

class StripClusterParameterEstimator;

template <class ResultingHit>
class TrackingRecHitProjector {
 public:

  typedef  TransientTrackingRecHit::RecHitPointer      RecHitPointer;


  RecHitPointer project( const TrackingRecHit& hit,
			 const GeomDet& det, 
			 const TrajectoryStateOnSurface& ts, const StripClusterParameterEstimator* cpe) const {
    
    GlobalVector gdir = ts.globalParameters().momentum();
    return project(hit, det, gdir, cpe);
  }
  RecHitPointer project( const TrackingRecHit& hit,
			 const GeomDet& det, const StripClusterParameterEstimator* cpe) const {
    GlobalVector gdir = hit.globalPosition() - GlobalPoint(0,0,0);
    return project(hit, det, gdir,cpe);
  }


  RecHitPointer project( const TransientTrackingRecHit& hit,
			 const GeomDet& det, 
			 const TrajectoryStateOnSurface& ts) const {

    GlobalVector gdir = ts.globalParameters().momentum();
    return project(hit, det, gdir);
  }
  RecHitPointer project( const TransientTrackingRecHit& hit,
			 const GeomDet& det) const {
    GlobalVector gdir = hit.globalPosition() - GlobalPoint(0,0,0);
    return project(hit, det, gdir);
  }

  RecHitPointer project( const TransientTrackingRecHit& hit,
			 const GeomDet& det,
			 const GlobalVector & gdir) const {
    const BoundPlane& gluedPlane = det.surface();
    const BoundPlane& hitPlane = hit.det()->surface();

    // check if the planes are parallel
    //const float epsilon = 1.e-7; // corresponds to about 0.3 miliradian but cannot be reduced
                                 // because of float precision

    //if (fabs(gluedPlane.normalVector().dot( hitPlane.normalVector())) < 1-epsilon) {
    //       std::cout << "TkGluedMeasurementDet plane not parallel to DetUnit plane: dot product is " 
    // 	   << gluedPlane.normalVector().dot( hitPlane.normalVector()) << endl;
    // FIXME: throw the appropriate exception here...
    //throw MeasurementDetException("TkGluedMeasurementDet plane not parallel to DetUnit plane");
    //}

    double delta = gluedPlane.localZ( hitPlane.position());
    LocalVector ldir = gluedPlane.toLocal(gdir);
    LocalPoint lhitPos = gluedPlane.toLocal( hit.globalPosition());
    LocalPoint projectedHitPos = lhitPos - ldir * delta/ldir.z();

    LocalVector hitXAxis = gluedPlane.toLocal( hitPlane.toGlobal( LocalVector(1,0,0)));
    LocalError hitErr = hit.localPositionError();
    if (gluedPlane.normalVector().dot( hitPlane.normalVector()) < 0) {
      // the two planes are inverted, and the correlation element must change sign
      hitErr = LocalError( hitErr.xx(), -hitErr.xy(), hitErr.yy());
    }
    LocalError rotatedError = hitErr.rotate( hitXAxis.x(), hitXAxis.y());
    
    return ResultingHit::build( projectedHitPos, rotatedError, &det, hit.det(), hit);
  }


  RecHitPointer project( const TrackingRecHit& hit,
			 const GeomDet& det,
			 const GlobalVector & gdir, const StripClusterParameterEstimator* cpe) const {
    const BoundPlane& gluedPlane = det.surface();
    const BoundPlane& hitPlane = hit.det()->surface();

    // check if the planes are parallel
    //const float epsilon = 1.e-7; // corresponds to about 0.3 miliradian but cannot be reduced
                                 // because of float precision

    //if (fabs(gluedPlane.normalVector().dot( hitPlane.normalVector())) < 1-epsilon) {
    //       std::cout << "TkGluedMeasurementDet plane not parallel to DetUnit plane: dot product is " 
    // 	   << gluedPlane.normalVector().dot( hitPlane.normalVector()) << endl;
    // FIXME: throw the appropriate exception here...
    //throw MeasurementDetException("TkGluedMeasurementDet plane not parallel to DetUnit plane");
    //}

    double delta = gluedPlane.localZ( hitPlane.position());
    LocalVector ldir = gluedPlane.toLocal(gdir);
    LocalPoint lhitPos = gluedPlane.toLocal( hit.globalPosition());
    LocalPoint projectedHitPos = lhitPos - ldir * delta/ldir.z();

    LocalVector hitXAxis = gluedPlane.toLocal( hitPlane.toGlobal( LocalVector(1,0,0)));
    LocalError hitErr = hit.localPositionError();
    if (gluedPlane.normalVector().dot( hitPlane.normalVector()) < 0) {
      // the two planes are inverted, and the correlation element must change sign
      hitErr = LocalError( hitErr.xx(), -hitErr.xy(), hitErr.yy());
    }
    LocalError rotatedError = hitErr.rotate( hitXAxis.x(), hitXAxis.y());
    
    return ResultingHit::build( projectedHitPos, rotatedError, &det, hit.det(), hit, cpe);
  }

};

inline
std::pair<LocalPoint,LocalError> projectedPos(const TrackingRecHit& hit,
                         const GeomDet& det,
                         const GlobalVector & gdir, const StripClusterParameterEstimator* cpe) {
    const BoundPlane& gluedPlane = det.surface();
    const BoundPlane& hitPlane = hit.det()->surface();

    // check if the planes are parallel
    //const float epsilon = 1.e-7; // corresponds to about 0.3 miliradian but cannot be reduced
                                 // because of float precision

    //if (fabs(gluedPlane.normalVector().dot( hitPlane.normalVector())) < 1-epsilon) {
    //       std::cout << "TkGluedMeasurementDet plane not parallel to DetUnit plane: dot product is "
    //     << gluedPlane.normalVector().dot( hitPlane.normalVector()) << endl;
    // FIXME: throw the appropriate exception here...
    //throw MeasurementDetException("TkGluedMeasurementDet plane not parallel to DetUnit plane");
    //}

    double delta = gluedPlane.localZ( hitPlane.position());
    LocalVector ldir = gluedPlane.toLocal(gdir);
    LocalPoint lhitPos = gluedPlane.toLocal( hit.globalPosition());
    LocalPoint projectedHitPos = lhitPos - ldir * delta/ldir.z();                  

    LocalVector hitXAxis = gluedPlane.toLocal( hitPlane.toGlobal( LocalVector(1,0,0)));
    LocalError hitErr = hit.localPositionError();
    if (gluedPlane.normalVector().dot( hitPlane.normalVector()) < 0) {
      // the two planes are inverted, and the correlation element must change sign
      hitErr = LocalError( hitErr.xx(), -hitErr.xy(), hitErr.yy());
    }
    LocalError rotatedError = hitErr.rotate( hitXAxis.x(), hitXAxis.y());

    return std::make_pair(projectedHitPos, rotatedError);
}

inline
std::pair<LocalPoint,LocalError> projectedPos(const TrackingRecHit& hit,
                         const GeomDet& det,
                         const TrajectoryStateOnSurface& ts, const StripClusterParameterEstimator* cpe) {

    GlobalVector gdir = ts.globalParameters().momentum();
    return projectedPos(hit, det, gdir, cpe);
}

#endif
