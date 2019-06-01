#ifndef TrackingTools_GeomPropagators_SmartPropagatorESProducer_H
#define TrackingTools_GeomPropagators_SmartPropagatorESProducer_H

/** \class SmartPropagatorESProducer
 *  ES producer needed to put the SmartPropagator inside the EventSetup
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/ESProducer.h"

#include "TrackingTools/GeomPropagators/interface/SmartPropagator.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include <memory>

namespace edm {
  class ParameterSet;
}

class TrackingComponentsRecord;

class SmartPropagatorESProducer : public edm::ESProducer {
public:
  /// Constructor
  SmartPropagatorESProducer(const edm::ParameterSet &);

  /// Destructor
  ~SmartPropagatorESProducer() override;

  // Operations
  std::unique_ptr<Propagator> produce(const TrackingComponentsRecord &);

private:
  PropagationDirection thePropagationDirection;
  std::string theTrackerPropagatorName;
  std::string theMuonPropagatorName;
  double theEpsilon;
};

#endif
