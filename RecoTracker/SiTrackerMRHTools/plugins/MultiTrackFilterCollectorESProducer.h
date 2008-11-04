#ifndef RecoLocalTracker_ESProducers_MultiTrackFilterCollectorESProducer_h
#define RecoLocalTracker_ESProducers_MultiTrackFilterCollectorESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/Record/interface/MultiRecHitRecord.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiRecHitCollector.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiTrackFilterHitCollector.h"
#include <boost/shared_ptr.hpp>

class  MultiTrackFilterCollectorESProducer: public edm::ESProducer{
 public:
  MultiTrackFilterCollectorESProducer(const edm::ParameterSet & p);
  virtual ~MultiTrackFilterCollectorESProducer(); 
  boost::shared_ptr<MultiTrackFilterHitCollector> produce(const MultiRecHitRecord &);
 private:
  boost::shared_ptr<MultiTrackFilterHitCollector> _collector;
  edm::ParameterSet pset_;
};


#endif
