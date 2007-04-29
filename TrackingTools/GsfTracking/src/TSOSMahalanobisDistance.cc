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

  AlgebraicVector5 mu1 = tsos1.localParameters().vector();
  const AlgebraicSymMatrix55 & V1 = tsos1.localError().matrix();
  AlgebraicVector5 mu2 = tsos2.localParameters().vector();
  const AlgebraicSymMatrix55 & V2 = tsos2.localError().matrix();

  int ierr;
  AlgebraicSymMatrix55 Vsum = V1 + V2;
  AlgebraicSymMatrix55 VsumInverse = Vsum.Inverse(ierr);
  AlgebraicVector5 mudiff = mu1 - mu2;

  double dist = ROOT::Math::Similarity(mudiff, VsumInverse);  //FIXME:: Optimize

  return dist;
}
