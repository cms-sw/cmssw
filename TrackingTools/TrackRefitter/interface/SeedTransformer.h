#ifndef SeedTransformer_H
#define SeedTransformer_H

/** \class SeedTransformer
 *  Description: this class takes a TrajectorySeed,
 *  fits its RecHits and returns a vector of Trajectories.
 *  If the fit fails, the returned vector is empty.
 *
 *  \author D. Trocino - University and INFN Torino
 */

// Base class header
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"

#include <vector>

namespace edm {
  class ParameterSet;
  class EventSetup;
  class ConsumesCollector;
}  // namespace edm
class Trajectory;
class TrajectorySeed;
class TrajectoryStateOnSurface;
class GlobalTrackingGeometry;
class MagneticField;
class TransientTrackingRecHitBuilder;
class Propagator;
class GlobalTrackingGeometryRecord;
class IdealMagneticFieldRecord;
class TransientRecHitRecord;
class TrackingComponentsRecord;

class SeedTransformer {
public:
  /// Constructor
  SeedTransformer(const edm::ParameterSet&, edm::ConsumesCollector);

  /// Destructor
  virtual ~SeedTransformer();

  // Operations
  /// Set the services needed by the SeedTransformer
  void setServices(const edm::EventSetup&);

  /// Performs the fit
  std::vector<Trajectory> seedTransform(const TrajectorySeed&) const;
  TrajectoryStateOnSurface seedTransientState(const TrajectorySeed&) const;

protected:
private:
  const GlobalTrackingGeometry* theTrackingGeometry;
  const MagneticField* theMagneticField;
  const TrajectoryFitter* theFitter;
  const TransientTrackingRecHitBuilder* theMuonRecHitBuilder;
  const Propagator* thePropagator;

  edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> theTrackingGeometryToken;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> theMagneticFieldToken;
  edm::ESGetToken<TrajectoryFitter, TrajectoryFitter::Record> theFitterToken;
  edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> theMuonRecHitBuilderToken;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> thePropagatorToken;

  /// Minimum number of RecHits required to perform the fit
  unsigned int nMinRecHits;

  /// Error rescale factor
  double errorRescale;

  bool useSubRecHits;
};
#endif
