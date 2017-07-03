#ifndef TrackingTools_ESProducers_TrackingRecHitPropagatorESProducer_h
#define TrackingTools_ESProducers_TrackingRecHitPropagatorESProducer_h


#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/TrackingRecHitPropagator.h"
#include <memory>

class  TrackingRecHitPropagatorESProducer: public edm::ESProducer{
 public:
  TrackingRecHitPropagatorESProducer(const edm::ParameterSet & p);
  ~TrackingRecHitPropagatorESProducer() override; 
  std::shared_ptr<TrackingRecHitPropagator> produce(const TrackingComponentsRecord&);
 private:
  std::shared_ptr<TrackingRecHitPropagator> theHitPropagator;
  edm::ParameterSet pset_;
};


#endif




