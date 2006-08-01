#include "TrackingTools/GsfTracking/interface/TSOSKullbackLeiblerDistance.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

double TSOSKullbackLeiblerDistance::operator() (const TrajectoryStateOnSurface& tsos1, 
						const TrajectoryStateOnSurface& tsos2) const {

  if (&tsos1.surface() != &tsos2.surface()) {
    edm::LogError("TSOSKullbackLeiblerDistance") 
      << "Trying to calculate distance between components defined "
      << "at different surfaces - returning zero!";
    return 0.;
  }

  AlgebraicVector mu1 = tsos1.localParameters().vector();
  AlgebraicSymMatrix V1 = tsos1.localError().matrix();
  AlgebraicVector mu2 = tsos2.localParameters().vector();
  AlgebraicSymMatrix V2 = tsos2.localError().matrix();

  int ierr;
  AlgebraicSymMatrix G1 = V1.inverse(ierr);
  AlgebraicSymMatrix G2 = V2.inverse(ierr);
  AlgebraicVector mudiff = mu1 - mu2;
  AlgebraicSymMatrix Vdiff = V1 - V2;
  AlgebraicSymMatrix Gdiff = G2 - G1;
  AlgebraicSymMatrix Gsum = G1 + G2;

  double dist = (Vdiff * Gdiff).trace() + Gsum.similarity(mudiff);

  return dist;
}
