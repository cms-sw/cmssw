#include "TrackingTools/Producers/interface/CompositeTrajectoryFilterESProducer.h"

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilterFactory.h"
#include "TrackingTools/TrajectoryFiltering/interface/CompositeTrajectoryFilter.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

CompositeTrajectoryFilterESProducer::CompositeTrajectoryFilterESProducer(const edm::ParameterSet& iConfig)
{
  componentName = iConfig.getParameter<std::string>("ComponentName");
  filterNames = iConfig.getParameter<std::vector<std::string> >("filterNames");
  
  edm::LogInfo("CompositeTrajectoryFilterESProducer")<<"configured to produce: CompositeTrajectoryFilterESProducer"
						     <<" with name: "<<componentName;
  setWhatProduced(this, componentName);
}


CompositeTrajectoryFilterESProducer::~CompositeTrajectoryFilterESProducer(){}

CompositeTrajectoryFilterESProducer::ReturnType
CompositeTrajectoryFilterESProducer::produce(const TrajectoryFilter::Record & record)
{
   using namespace edm::es;
   edm::LogInfo("CompositeTrajectoryFilterESProducer")<<"producing: "<<componentName<<" of type: CompositeTrajectoryFilterESProducer";

   std::vector<const TrajectoryFilter*> filters;
   edm::ESHandle<TrajectoryFilter> aFilterH;
   for (unsigned int i=0;i!=filterNames.size();++i)
     {
       record.get(filterNames[i], aFilterH);
       edm::LogInfo("CompositeTrajectoryFilterESProducer")<<"adding: "<<filterNames[i];
       filters.push_back(aFilterH.product());
     }

   CompositeTrajectoryFilterESProducer::ReturnType aFilter( new CompositeTrajectoryFilter(filters));
   
   return aFilter ;
}

