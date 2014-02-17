#ifndef TrajectoryInValidHit_H
#define TrajectoryInValidHit_H

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

class Topology;
class TransientTrackingRecHit;
class StripTopology;
class PixelTopology;
class TrajectoryInValidHit {
public:

  TrajectoryInValidHit( const TrajectoryMeasurement&, const TrackerGeometry * tracker);

  double localRPhiX() const;
  double localRPhiY() const;
  double localStereoX() const;
  double localStereoY() const;
  double localErrorX() const;
  double localErrorY() const;
 
  double localZ() const;

  double globalX() const;
  double globalY() const;
  double globalZ() const;
  bool InValid() const; 
 
private:

  bool IsInvHit;

  typedef TrajectoryStateOnSurface TSOS;

  TSOS theCombinedPredictedState;
  float RPhilocX_temp,RPhilocY_temp, StereolocX_temp,StereolocY_temp;
  float RPhilocX,RPhilocY, StereolocX,StereolocY;

  //  const TransientTrackingRecHit* theHit;
  ConstReferenceCountingPointer<TransientTrackingRecHit> theHit;
  LocalPoint project(const GeomDet *det,const GeomDet* projdet,LocalPoint position,LocalVector trackdirection)const;

};

#endif
