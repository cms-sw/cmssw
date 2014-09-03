#ifndef RecoLocalTracker_ESProducers_SiTrackerMultiRecHitUpdatorESProducer_h
#define RecoLocalTracker_ESProducers_ESProducers_SiTrackerMultiRecHitUpdatorESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/Record/interface/MultiRecHitRecord.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdator.h"

#include <boost/shared_ptr.hpp>

class  SiTrackerMultiRecHitUpdatorESProducer: public edm::ESProducer{
 public:
  SiTrackerMultiRecHitUpdatorESProducer(const edm::ParameterSet & p);
  virtual ~SiTrackerMultiRecHitUpdatorESProducer(); 
  boost::shared_ptr<SiTrackerMultiRecHitUpdator> produce(const MultiRecHitRecord &);
 private:
  boost::shared_ptr<SiTrackerMultiRecHitUpdator> _updator;
  edm::ParameterSet pset_;
};


#endif




