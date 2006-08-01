#include "TrackingTools/GsfTracking/interface/TSOSMahalanobisDistance.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

double TSOSMahalanobisDistance::operator() (const TrajectoryStateOnSurface& tsos1, 
					    const TrajectoryStateOnSurface& tsos2) const {

  if (&tsos1.surface() != &tsos2.surface()) {
    edm::LogError("TSOSMahalanobisDistance") 
      << "Trying to calculate distance between components defined "
      << "at different surfaces - returning zero!" ;
    return 0.;
  }

  AlgebraicVector mu1 = tsos1.localParameters().vector();
  AlgebraicSymMatrix V1 = tsos1.localError().matrix();
  AlgebraicVector mu2 = tsos2.localParameters().vector();
  AlgebraicSymMatrix V2 = tsos2.localError().matrix();

  int ierr;
  AlgebraicSymMatrix VsumInverse = (V1 + V2).inverse(ierr);
  AlgebraicVector mudiff = mu1 - mu2;

  double dist = VsumInverse.similarity(mudiff);

  return dist;
}
