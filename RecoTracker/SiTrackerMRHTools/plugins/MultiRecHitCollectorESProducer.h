#ifndef RecoLocalTracker_ESProducers_MultiRecHitCollectorESProducer_h
#define RecoLocalTracker_ESProducers_ESProducers_MultiRecHitCollectorESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/Record/interface/MultiRecHitRecord.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiRecHitCollector.h"

#include <boost/shared_ptr.hpp>

class  MultiRecHitCollectorESProducer: public edm::ESProducer{
 public:
  MultiRecHitCollectorESProducer(const edm::ParameterSet & p);
  virtual ~MultiRecHitCollectorESProducer(); 
  boost::shared_ptr<MultiRecHitCollector> produce(const MultiRecHitRecord &);
 private:
  boost::shared_ptr<MultiRecHitCollector> _collector;
  edm::ParameterSet pset_;
};


#endif




