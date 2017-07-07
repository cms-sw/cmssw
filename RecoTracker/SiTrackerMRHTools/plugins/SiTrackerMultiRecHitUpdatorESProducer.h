#ifndef RecoLocalTracker_ESProducers_SiTrackerMultiRecHitUpdatorESProducer_h
#define RecoLocalTracker_ESProducers_SiTrackerMultiRecHitUpdatorESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/Record/interface/MultiRecHitRecord.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdator.h"

#include <memory>

class  SiTrackerMultiRecHitUpdatorESProducer: public edm::ESProducer{
 public:
  SiTrackerMultiRecHitUpdatorESProducer(const edm::ParameterSet & p);
  ~SiTrackerMultiRecHitUpdatorESProducer() override; 
  std::shared_ptr<SiTrackerMultiRecHitUpdator> produce(const MultiRecHitRecord &);
 private:
  std::shared_ptr<SiTrackerMultiRecHitUpdator> _updator;
  edm::ParameterSet pset_;
};


#endif // RecoLocalTracker_ESProducers_SiTrackerMultiRecHitUpdatorESProducer_h
