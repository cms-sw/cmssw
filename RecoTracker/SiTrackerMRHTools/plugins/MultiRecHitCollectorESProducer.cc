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

MultiRecHitCollectorESProducer::MultiRecHitCollectorESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

MultiRecHitCollectorESProducer::~MultiRecHitCollectorESProducer() {}

boost::shared_ptr<MultiRecHitCollector> 
MultiRecHitCollectorESProducer::produce(const MultiRecHitRecord& iRecord){
  std::string mode = "Grouped";
  if (pset_.getParameter<std::string>("Mode")=="Simple") mode = "Simple"; 

  std::string mrhupdator             = pset_.getParameter<std::string>("MultiRecHitUpdator"); 
  std::string propagatorAlongName    = pset_.getParameter<std::string>("propagatorAlong");
  std::string estimatorName          = pset_.getParameter<std::string>("estimator"); 
  std::string measurementTrackerName = pset_.getParameter<std::string>("MeasurementTrackerName");

  ESHandle<SiTrackerMultiRecHitUpdator> mrhuhandle;
  iRecord.get(mrhupdator, mrhuhandle);
  ESHandle<Propagator>	propagatorhandle;
  iRecord.getRecord<CkfComponentsRecord>().getRecord<TrackingComponentsRecord>().get(propagatorAlongName, propagatorhandle);
  ESHandle<Chi2MeasurementEstimatorBase> estimatorhandle;
  iRecord.getRecord<CkfComponentsRecord>().getRecord<TrackingComponentsRecord>().get(estimatorName, estimatorhandle);
  ESHandle<MeasurementTracker> measurementhandle;
  iRecord.getRecord<CkfComponentsRecord>().get(measurementTrackerName, measurementhandle);	 
 
  if (mode == "Grouped"){
	std::string propagatorOppositeName = pset_.getParameter<std::string>("propagatorOpposite");  
	ESHandle<Propagator>  propagatorOppositehandle;
  	iRecord.getRecord<CkfComponentsRecord>().getRecord<TrackingComponentsRecord>().get(propagatorOppositeName, propagatorOppositehandle); 
  	_collector  = boost::shared_ptr<MultiRecHitCollector>(new GroupedDAFHitCollector(measurementhandle.product(),
									         mrhuhandle.product(),
										 estimatorhandle.product(),
										 propagatorhandle.product(),
										 propagatorOppositehandle.product()));
  } else {
	_collector  = boost::shared_ptr<MultiRecHitCollector>(new SimpleDAFHitCollector(measurementhandle.product(),
                                                                                 mrhuhandle.product(),
                                                                                 estimatorhandle.product(),
                                                                                 propagatorhandle.product()));
  }
  	
  return _collector;
}


