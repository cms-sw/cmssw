#include "TrackingTools/GsfTools/interface/DistanceBetweenComponentsESProducer.h"

#include "TrackingTools/GsfTools/interface/KullbackLeiblerDistance.h"
#include "TrackingTools/GsfTools/interface/MahalanobisDistance.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

DistanceBetweenComponentsESProducer::DistanceBetweenComponentsESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

DistanceBetweenComponentsESProducer::~DistanceBetweenComponentsESProducer() {}

boost::shared_ptr<DistanceBetweenComponents> 
DistanceBetweenComponentsESProducer::produce(const TrackingComponentsRecord & iRecord){ 

  std::string distName = pset_.getParameter<std::string>("DistanceMeasure");
  
  boost::shared_ptr<DistanceBetweenComponents> distance;
  if ( distName == "KullbackLeibler" )
    distance = boost::shared_ptr<DistanceBetweenComponents>(new KullbackLeiblerDistance());
  else if ( distName == "Mahalanobis" )
    distance = boost::shared_ptr<DistanceBetweenComponents>(new MahalanobisDistance());
  
  return distance;
}


