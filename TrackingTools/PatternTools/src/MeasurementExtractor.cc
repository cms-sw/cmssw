#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"

AlgebraicVector MeasurementExtractor::measuredParameters(const TrackingRecHit& hit) {
  AlgebraicVector par5( asHepVector( theTSoS.localParameters().vector() ) );
  AlgebraicMatrix H( hit.projectionMatrix());
  return H*par5;
}

AlgebraicSymMatrix MeasurementExtractor::measuredError(const TrackingRecHit& hit) {
  AlgebraicSymMatrix err5( asHepMatrix( theTSoS.localError().matrix() ) );
  AlgebraicMatrix H( hit.projectionMatrix());
  //  return AlgebraicSymMatrix( H * err5 * H.T());
  return err5.similarity(H);
}
