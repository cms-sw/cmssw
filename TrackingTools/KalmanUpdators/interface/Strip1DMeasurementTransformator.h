#ifndef CD_Strip1DMeasurementTransformator_H_
#define CD_Strip1DMeasurementTransformator_H_

/** \class Strip1DMeasurementTransformator
 *  Helper class for accessing the RecHit and the TrajectoryState parameters
 *  and errors in the measurement frame. The RecHits are treated as
 *  1-dimensional, the second coordinate is ignored. Ported from ORCA.
 *
 *  \author todorov, cerati
 */

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

class Strip1DMeasurementTransformator {

private:

  typedef TrajectoryStateOnSurface TSOS;
  typedef AlgebraicSymMatrix55 ASM55;
  typedef AlgebraicMatrix15 AM15;
  typedef AlgebraicVector5 AV5;

public:

  Strip1DMeasurementTransformator(const TSOS& aTsos, const TrackingRecHit& aHit);
  
  ~Strip1DMeasurementTransformator() {}

  double hitParameters() const;
  AV5 trajectoryParameters() const;
  double projectedTrajectoryParameters() const;
  double hitError() const;
  const ASM55 & trajectoryError() const;
  double projectedTrajectoryError() const;
  AM15 projectionMatrix() const;

  const TrackingRecHit& hit() const {return theRecHit;}
  const TSOS& state() const {return theState;}
  /// return ideal topology, as if the sensor were flat
  const StripTopology* idealTopology() const {return theIdealTopology;}
  /// return real topology, taking sensor deformations into account
  const StripTopology* topology() const {return theTopology;}

private:

  const TrackingRecHit& theRecHit;
  TSOS theState;
  const StripTopology *theTopology, *theIdealTopology;

  void init();
};

#endif //CD_Strip1DMeasurementTransformator_H_
