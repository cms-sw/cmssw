#ifndef RecoLocalTracker_ESProducers_SiTrackerMultiRecHitUpdatorMTFESProducer_h
#define RecoLocalTracker_ESProducers_SiTrackerMultiRecHitUpdatorMTFESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/Record/interface/MultiRecHitRecord.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdatorMTF.h"

#include <boost/shared_ptr.hpp>

class  SiTrackerMultiRecHitUpdatorMTFESProducer: public edm::ESProducer{
 public:
  SiTrackerMultiRecHitUpdatorMTFESProducer(const edm::ParameterSet & p);
  virtual ~SiTrackerMultiRecHitUpdatorMTFESProducer(); 
  boost::shared_ptr<SiTrackerMultiRecHitUpdatorMTF> produce(const MultiRecHitRecord &);
 private:
  boost::shared_ptr<SiTrackerMultiRecHitUpdatorMTF> _updator;
  edm::ParameterSet pset_;
};


#endif




