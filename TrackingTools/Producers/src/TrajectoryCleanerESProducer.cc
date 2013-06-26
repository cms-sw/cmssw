#include "TrackingTools/Producers/interface/TrajectoryCleanerESProducer.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerFactory.h"

TrajectoryCleanerESProducer::TrajectoryCleanerESProducer(const edm::ParameterSet& iConfig)
{
  theComponentName = iConfig.getParameter<std::string>("ComponentName");
  theComponentType = iConfig.getParameter<std::string>("ComponentType");

  theConfig = iConfig;
  setWhatProduced(this, theComponentName);
}


TrajectoryCleanerESProducer::~TrajectoryCleanerESProducer(){}

// ------------ method called to produce the data  ------------
TrajectoryCleanerESProducer::ReturnType
TrajectoryCleanerESProducer::produce(const  TrackingComponentsRecord & iRecord)
{
   using namespace edm::es;
   
   ReturnType tc(TrajectoryCleanerFactory::get()->create(theComponentType, theConfig));
   return tc;
}
