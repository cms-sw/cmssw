#ifndef TrackingTools_GeomPropagators_BeamHaloPropagatorESProducer_H
#define TrackingTools_GeomPropagators_BeamHaloPropagatorESProducer_H

/** \class BeamHaloPropagatorESProducer
 *  ES producer needed to put the BeamHaloPropagator inside the EventSetup
 *
 *  \author Jean-Roch VLIMANT UCSB
 */

#include "FWCore/Framework/interface/ESProducer.h"

#include "TrackingTools/GeomPropagators/interface/BeamHaloPropagator.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include <memory>

namespace edm {
  class ParameterSet;
}

class TrackingComponentsRecord;

class BeamHaloPropagatorESProducer : public edm::ESProducer {
public:
  /// Constructor
  BeamHaloPropagatorESProducer(const edm::ParameterSet &);

  /// Destructor
  ~BeamHaloPropagatorESProducer() override;

  // Operations
  std::unique_ptr<Propagator> produce(const TrackingComponentsRecord &);

private:
  PropagationDirection thePropagationDirection;
  std::string myname;
  std::string theEndCapTrackerPropagatorName;
  std::string theCrossingTrackerPropagatorName;
};

#endif
