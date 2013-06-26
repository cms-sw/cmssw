#ifndef TrackPropagation_ESProducers_AnalyticalPropagatorESProducer_h
#define TrackPropagation_ESProducers_AnalyticalPropagatorESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include <boost/shared_ptr.hpp>

class  SteppingHelixPropagatorESProducer: public edm::ESProducer{
 public:
  SteppingHelixPropagatorESProducer(const edm::ParameterSet & p);
  virtual ~SteppingHelixPropagatorESProducer(); 
  boost::shared_ptr<Propagator> produce(const TrackingComponentsRecord &);
 private:
  boost::shared_ptr<Propagator> _propagator;
  edm::ParameterSet pset_;
};


#endif
