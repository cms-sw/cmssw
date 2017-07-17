#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdator.h"
#include "RecoTracker/SiTrackerMRHTools/plugins/MultiRecHitCollectorESProducer.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/SiTrackerMRHTools/interface/GroupedDAFHitCollector.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SimpleDAFHitCollector.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;

MultiRecHitCollectorESProducer::MultiRecHitCollectorESProducer(const edm::ParameterSet& iConfig) 
{
  std::string myname = iConfig.getParameter<std::string>("ComponentName");
  setConf(iConfig);
  setWhatProduced(this,myname);
}

MultiRecHitCollectorESProducer::~MultiRecHitCollectorESProducer() {}

std::shared_ptr<MultiRecHitCollector> 
MultiRecHitCollectorESProducer::produce(const MultiRecHitRecord& iRecord){
  std::string mode = "Grouped";
  if (conf_.getParameter<std::string>("Mode")=="Simple") mode = "Simple"; 

  std::string mrhupdator             = conf_.getParameter<std::string>("MultiRecHitUpdator"); 
  std::string propagatorAlongName    = conf_.getParameter<std::string>("propagatorAlong");
  std::string estimatorName          = conf_.getParameter<std::string>("estimator"); 
  std::string measurementTrackerName = conf_.getParameter<std::string>("MeasurementTrackerName");
  bool debug = conf_.getParameter<bool>("Debug");


  ESHandle<SiTrackerMultiRecHitUpdator> mrhuhandle;
  iRecord.get(mrhupdator, mrhuhandle);
  ESHandle<Propagator>	propagatorhandle;
  iRecord.getRecord<CkfComponentsRecord>().getRecord<TrackingComponentsRecord>().get(propagatorAlongName, propagatorhandle);
  ESHandle<Chi2MeasurementEstimatorBase> estimatorhandle;
  iRecord.getRecord<CkfComponentsRecord>().getRecord<TrackingComponentsRecord>().get(estimatorName, estimatorhandle);
  ESHandle<MeasurementTracker> measurementhandle;
  iRecord.getRecord<CkfComponentsRecord>().get(measurementTrackerName, measurementhandle);
  ESHandle<TrackerTopology> trackerTopologyHandle;
  iRecord.getRecord<CkfComponentsRecord>().getRecord<TrackerTopologyRcd>().get(trackerTopologyHandle);
 
  if (mode == "Grouped"){
	std::string propagatorOppositeName = conf_.getParameter<std::string>("propagatorOpposite");  
	ESHandle<Propagator>  propagatorOppositehandle;
  	iRecord.getRecord<CkfComponentsRecord>().getRecord<TrackingComponentsRecord>().get(propagatorOppositeName, propagatorOppositehandle); 
  	collector_ = std::make_shared<GroupedDAFHitCollector>(measurementhandle.product(),
						         mrhuhandle.product(),
							 estimatorhandle.product(),
							 propagatorhandle.product(),
							 propagatorOppositehandle.product(), debug);
  } 
  else {
	collector_ = std::make_shared<SimpleDAFHitCollector>(trackerTopologyHandle.product(),
                                                             measurementhandle.product(),
                                                             mrhuhandle.product(),
                                                             estimatorhandle.product(),
                                                             propagatorhandle.product(), debug);
  }
  	
  return collector_;

}


