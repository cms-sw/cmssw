#ifndef TrackingTools_ESProducers_TrackingRecHitPropagatorESProducer_h
#define TrackingTools_ESProducers_TrackingRecHitPropagatorESProducer_h


#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/TrackingRecHitPropagator.h"
#include <boost/shared_ptr.hpp>

class  TrackingRecHitPropagatorESProducer: public edm::ESProducer{
 public:
  TrackingRecHitPropagatorESProducer(const edm::ParameterSet & p);
  virtual ~TrackingRecHitPropagatorESProducer(); 
  boost::shared_ptr<TrackingRecHitPropagator> produce(const TrackingComponentsRecord&);
 private:
  boost::shared_ptr<TrackingRecHitPropagator> theHitPropagator;
  edm::ParameterSet pset_;
};


#endif




