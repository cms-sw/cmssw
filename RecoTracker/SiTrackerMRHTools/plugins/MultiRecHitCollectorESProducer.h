#ifndef RecoLocalTracker_ESProducers_MultiRecHitCollectorESProducer_h
#define RecoLocalTracker_ESProducers_MultiRecHitCollectorESProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/Record/interface/MultiRecHitRecord.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiRecHitCollector.h"

#include <memory>

class  MultiRecHitCollectorESProducer: public edm::ESProducer{
 public:
  MultiRecHitCollectorESProducer(const edm::ParameterSet& iConfig);
  ~MultiRecHitCollectorESProducer() override; 
  std::shared_ptr<MultiRecHitCollector> produce(const MultiRecHitRecord &);

  // Set parameter set
  void setConf(const edm::ParameterSet& conf){ conf_ = conf; }
 
 private:
  std::shared_ptr<MultiRecHitCollector> collector_;
  edm::ParameterSet conf_;

};


#endif // RecoLocalTracker_ESProducers_MultiRecHitCollectorESProducer_h
