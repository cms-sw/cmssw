#include "TrackingTools/GsfTracking/interface/CloseComponentsTSOSMergerESProducer.h"

#include "TrackingTools/GsfTracking/interface/CloseComponentsTSOSMerger.h"
#include "TrackingTools/GsfTracking/interface/TSOSDistanceBetweenComponents.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

CloseComponentsTSOSMergerESProducer::CloseComponentsTSOSMergerESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

CloseComponentsTSOSMergerESProducer::~CloseComponentsTSOSMergerESProducer() {}

boost::shared_ptr<MultiTrajectoryStateMerger> 
CloseComponentsTSOSMergerESProducer::produce(const TrackingComponentsRecord & iRecord){ 

  int maxComp = pset_.getParameter<int>("MaxComponents");
  std::string distName = pset_.getParameter<std::string>("DistanceMeasure");
  
  edm::ESHandle<TSOSDistanceBetweenComponents> distProducer;
  iRecord.get(distName,distProducer);
  
  return 
    boost::shared_ptr<MultiTrajectoryStateMerger>(new CloseComponentsTSOSMerger(maxComp,
										distProducer.product()));
}


