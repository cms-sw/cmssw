#ifndef TrajectoryMeasurementResidual_H
#define TrajectoryMeasurementResidual_H

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"

class Topology;
class TransientTrackingRecHit;
class StripTopology;
class PixelTopology;
class TrajectoryMeasurementResidual {
public:

  TrajectoryMeasurementResidual( const TrajectoryMeasurement&);

  double localXResidual() const;
  double localYResidual() const;

  bool hasMeasurementFrame() const
  {return theMeasFrame;};
  double measurementXResidual() const;
  double measurementYResidual() const;
  double measurementXResidualInPitch() const;
  double measurementYResidualInPitch() const;

  double localXError() const;
  double localYError() const;
  double measurementXError() const;
  double measurementYError() const;
  double measurementXErrorInPitch() const;
  double measurementYErrorInPitch() const;

  LocalError localError() const;

private:

  typedef TrajectoryStateOnSurface TSOS;

  bool theMeasFrame;
  TSOS theCombinedPredictedState;
  //  const TransientTrackingRecHit* theHit;
  ConstReferenceCountingPointer<TransientTrackingRecHit> theHit;
  const Topology* theTopol;
  Measurement2DVector theMeasResidual;
  const MeasurementError *theMeasError;
  const StripTopology* theStripTopol;
  const PixelTopology* thePixelTopol;

  void checkMeas() const;
  double xPitch() const;
  double yPitch() const;

};

#endif
