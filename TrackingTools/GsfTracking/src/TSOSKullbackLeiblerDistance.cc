#include "TrackingTools/GsfTracking/interface/TSOSKullbackLeiblerDistance.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

inline double FIXME_trace(const AlgebraicMatrix55& m) { 
        return m(0,0)+m(1,1)+m(2,2)+m(3,3)+m(4,4);
}

double TSOSKullbackLeiblerDistance::operator() (const TrajectoryStateOnSurface& tsos1, 
						const TrajectoryStateOnSurface& tsos2) const {

  if (&tsos1.surface() != &tsos2.surface()) {
    edm::LogError("TSOSKullbackLeiblerDistance") 
      << "Trying to calculate distance between components defined "
      << "at different surfaces - returning zero!";
    return 0.;
  }

  AlgebraicVector5 mu1 = tsos1.localParameters().vector();
  const AlgebraicSymMatrix55 & V1 = tsos1.localError().matrix();
  AlgebraicVector5 mu2 = tsos2.localParameters().vector();
  const AlgebraicSymMatrix55 & V2 = tsos2.localError().matrix();

  int ierr;
  AlgebraicSymMatrix55 G1 = V1.Inverse(ierr);
  AlgebraicSymMatrix55 G2 = V2.Inverse(ierr);
  AlgebraicVector5 mudiff = mu1 - mu2;
  AlgebraicSymMatrix55 Vdiff = V1 - V2;
  AlgebraicSymMatrix55 Gdiff = G2 - G1;
  //AlgebraicSymMatrix55 Gsum = G1 + G2;

  double dist = FIXME_trace(Vdiff * Gdiff) + ROOT::Math::Similarity(mudiff, G1 + G2); ///FIXME: optimize!!

  return dist;
}
