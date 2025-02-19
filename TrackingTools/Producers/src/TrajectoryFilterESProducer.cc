#include "TrackingTools/Producers/interface/TrajectoryFilterESProducer.h"

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilterFactory.h"
//#include "TrackingTools/TrajectoryFiltering/interface/ClusterShapeTrajectoryFilter.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

TrajectoryFilterESProducer::TrajectoryFilterESProducer(const edm::ParameterSet& iConfig)
{
  componentName = iConfig.getParameter<std::string>("ComponentName");
  
  
  filterPset = iConfig.getParameter<edm::ParameterSet>("filterPset");
  componentType = filterPset.getParameter<std::string>("ComponentType");
  
  edm::LogInfo("TrajectoryFilterESProducer")<<"configured to produce: "<<componentType
					    <<" with name: "<<componentName;
      
  //this is a bit nasty, but still ok
  if (componentType.find("ClusterShape")==std::string::npos){
    setWhatProduced(this, componentName);
  }
  else{
    //    setWhatProduced(this, &TrajectoryFilterESProducer::produceClusterShapeFilter, edm::es::Label(componentName));
  }
}


TrajectoryFilterESProducer::~TrajectoryFilterESProducer(){}

TrajectoryFilterESProducer::ReturnType
TrajectoryFilterESProducer::produce(const TrajectoryFilter::Record &)
{
   using namespace edm::es;
   edm::LogInfo("TrajectoryFilterESProducer")<<"producing: "<<componentName<<" of type: "<<componentType;

   //produce the filter using the plugin factory
   TrajectoryFilterESProducer::ReturnType aFilter(TrajectoryFilterFactory::get()->create(componentType ,filterPset));
   
   return aFilter ;
}


/*
TrajectoryFilterESProducer::ReturnType
TrajectoryFilterESProducer::produceClusterShapeFilter(const TrajectoryFilter::Record &iRecord)
{
   using namespace edm::es;
   edm::LogInfo("TrajectoryFilterESProducer")<<"producing: "<<componentName<<" of type: "<<componentType;

   //retrieve magentic fiedl
   edm::ESHandle<MagneticField> field;
   iRecord.getRecord<IdealMagneticFieldRecord>().get(field);

   //retrieve geometry
   edm::ESHandle<TrackingGeometry> tracker;
   iRecord.getRecord<TrackerDigiGeometryRecord>().get(tracker);

   //produce the filter using the plugin factory
   TrajectoryFilterESProducer::ReturnType aFilter(new ClusterShapeTrajectoryFilter(tracker.product(),
										   field.product(),
										   filterPset.getParameter<int>("Mode"));
   return aFilter ;
}
*/
