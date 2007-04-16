#include "TrackingTools/GsfTracking/plugins/LargestWeightsTSOSMergerESProducer.h"

#include "TrackingTools/GsfTracking/interface/LargestWeightsTSOSMerger.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

LargestWeightsTSOSMergerESProducer::LargestWeightsTSOSMergerESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

LargestWeightsTSOSMergerESProducer::~LargestWeightsTSOSMergerESProducer() {}

boost::shared_ptr<MultiTrajectoryStateMerger> 
LargestWeightsTSOSMergerESProducer::produce(const TrackingComponentsRecord & iRecord){ 

  int maxComp = pset_.getParameter<int>("MaxComponents");
  bool combine = pset_.getParameter<bool>("CombineSmallestComponents");
  
  return 
    boost::shared_ptr<MultiTrajectoryStateMerger>(new LargestWeightsTSOSMerger(maxComp,
									       combine));
}


