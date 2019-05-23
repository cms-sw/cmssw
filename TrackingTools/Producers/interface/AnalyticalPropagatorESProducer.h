#ifndef TrackingTools_ESProducers_AnalyticalPropagatorESProducer_h
#define TrackingTools_ESProducers_AnalyticalPropagatorESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include <memory>

class AnalyticalPropagatorESProducer : public edm::ESProducer {
public:
  AnalyticalPropagatorESProducer(const edm::ParameterSet &p);
  ~AnalyticalPropagatorESProducer() override;
  std::unique_ptr<Propagator> produce(const TrackingComponentsRecord &);

private:
  edm::ParameterSet pset_;
};

#endif
