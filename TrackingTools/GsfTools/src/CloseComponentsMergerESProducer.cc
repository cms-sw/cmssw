#include "TrackingTools/GsfTools/interface/CloseComponentsMergerESProducer.h"

#include "TrackingTools/GsfTools/interface/CloseComponentsMerger.h"
#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

CloseComponentsMergerESProducer::CloseComponentsMergerESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

CloseComponentsMergerESProducer::~CloseComponentsMergerESProducer() {}

boost::shared_ptr<CloseComponentsMerger> 
CloseComponentsMergerESProducer::produce(const TrackingComponentsRecord & iRecord){ 

  int maxComp = pset_.getParameter<int>("MaxComponents");
  std::string distName = pset_.getParameter<std::string>("DistanceMeasure");
  
  edm::ESHandle<DistanceBetweenComponents> distProducer;
  iRecord.get(distName,distProducer);
  
  return 
    boost::shared_ptr<CloseComponentsMerger>(new CloseComponentsMerger(maxComp,
								       distProducer.product()));
}


