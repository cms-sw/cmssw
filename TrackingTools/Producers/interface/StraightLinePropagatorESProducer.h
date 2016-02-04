#ifndef TrackingTools_ESProducers_StraightLinePropagatorESProducer_h
#define TrackingTools_ESProducers_StraightLinePropagatorESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePropagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include <boost/shared_ptr.hpp>

class  StraightLinePropagatorESProducer: public edm::ESProducer{
 public:
  StraightLinePropagatorESProducer(const edm::ParameterSet & p);
  virtual ~StraightLinePropagatorESProducer(); 
  boost::shared_ptr<Propagator> produce(const TrackingComponentsRecord &);
 private:
  boost::shared_ptr<Propagator> _propagator;
  edm::ParameterSet pset_;
};


#endif




