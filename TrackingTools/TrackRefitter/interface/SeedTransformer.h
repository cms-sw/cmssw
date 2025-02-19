#ifndef SeedTransformer_H
#define SeedTransformer_H

/** \class SeedTransformer
 *  Description: this class takes a TrajectorySeed,
 *  fits its RecHits and returns a vector of Trajectories.
 *  If the fit fails, the returned vector is empty.
 *
 *  $Date: 2008/11/11 08:01:23 $
 *  $Revision: 1.4 $
 *  \author D. Trocino - University and INFN Torino
 */

// Base class header
#include "FWCore/Framework/interface/ESHandle.h"
#include <vector>

namespace edm {
  class ParameterSet;
  class EventSetup;
}
class Trajectory;
class TrajectorySeed;
class TrajectoryStateOnSurface;
class GlobalTrackingGeometry;
class MagneticField;
class TrajectoryFitter;
class TransientTrackingRecHitBuilder;
class Propagator;

class SeedTransformer {
public:
  /// Constructor
  SeedTransformer(const edm::ParameterSet&);

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
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  edm::ESHandle<MagneticField> theMagneticField;
  edm::ESHandle<TrajectoryFitter> theFitter;
  edm::ESHandle<TransientTrackingRecHitBuilder> theMuonRecHitBuilder;
  edm::ESHandle<Propagator> thePropagator;

  std::string theFitterName;
  std::string theMuonRecHitBuilderName;
  std::string thePropagatorName ;

  /// Minimum number of RecHits required to perform the fit
  unsigned int nMinRecHits;

  /// Error rescale factor
  double errorRescale;

  bool useSubRecHits;

};
#endif

