#ifndef CD_StripMeasurementTransformator_H_
#define CD_StripMeasurementTransformator_H_

/** \class StripMeasurementTransformator
 *  Helper class for accessing the RecHit and the TrajectoryState parameters
 *  and errors in the measurement frame. Ported from ORCA.
 *
 *  $Date: 2007/05/09 13:50:25 $
 *  $Revision: 1.4 $
 *  \author todorov, cerati
 */

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"


class StripMeasurementTransformator {

private:

  typedef TrajectoryStateOnSurface TSOS;
  typedef AlgebraicSymMatrix22 ASM22;
  typedef AlgebraicSymMatrix55 ASM55;
  typedef AlgebraicMatrix AM;
  typedef AlgebraicMatrix25 AM25;
  typedef AlgebraicVector5 AV5;
  typedef AlgebraicVector2 AV2;

public:

  StripMeasurementTransformator(const TSOS& aTsos, const TransientTrackingRecHit& aHit);
  
  ~StripMeasurementTransformator() {}

  AV2 hitParameters() const;
  AV5 trajectoryParameters() const;
  AV2 projectedTrajectoryParameters() const;
  ASM22 hitError() const;
  const ASM55 & trajectoryError() const;
  ASM22 projectedTrajectoryError() const;
  AM25 projectionMatrix() const;

  const TransientTrackingRecHit& hit() const {return theRecHit;}
  const TSOS& state() const {return theState;}
  const StripTopology* topology() const {return theTopology;}

private:

  const TransientTrackingRecHit& theRecHit;
  TSOS theState;
  const StripTopology* theTopology;

  void init();
};

#endif //CD_StripMeasurementTransformator_H_
