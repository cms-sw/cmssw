#ifndef TrackingTools_ESProducers_AnalyticalPropagatorESProducer_h
#define TrackingTools_ESProducers_AnalyticalPropagatorESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include <boost/shared_ptr.hpp>

class  AnalyticalPropagatorESProducer: public edm::ESProducer{
 public:
  AnalyticalPropagatorESProducer(const edm::ParameterSet & p);
  virtual ~AnalyticalPropagatorESProducer(); 
  boost::shared_ptr<Propagator> produce(const TrackingComponentsRecord &);
 private:
  boost::shared_ptr<Propagator> _propagator;
  edm::ParameterSet pset_;
};


#endif




