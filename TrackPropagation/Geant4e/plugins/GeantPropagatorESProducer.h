#ifndef TrackPropagators_ESProducers_GeantPropagatorESProducer_h
#define TrackPropagators_ESProducers_GeantPropagatorESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include <memory>

/*
 * GeantPropagatorESProducer
 *
 * Produces an Geant4ePropagator for track propagation
 *
 */

class GeantPropagatorESProducer : public edm::ESProducer {
public:
  GeantPropagatorESProducer(const edm::ParameterSet &p);
  ~GeantPropagatorESProducer() override;

  std::unique_ptr<Propagator> produce(const TrackingComponentsRecord &);

private:
  edm::ParameterSet pset_;
  double plimit_;
};

#endif
