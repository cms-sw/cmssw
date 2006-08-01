#include "TrackingTools/GsfTracking/interface/TSOSDistanceESProducer.h"

#include "TrackingTools/GsfTracking/interface/TSOSKullbackLeiblerDistance.h"
#include "TrackingTools/GsfTracking/interface/TSOSMahalanobisDistance.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

TSOSDistanceESProducer::TSOSDistanceESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

TSOSDistanceESProducer::~TSOSDistanceESProducer() {}

boost::shared_ptr<TSOSDistanceBetweenComponents> 
TSOSDistanceESProducer::produce(const TrackingComponentsRecord & iRecord){ 

  std::string distName = pset_.getParameter<std::string>("DistanceMeasure");
  
  boost::shared_ptr<TSOSDistanceBetweenComponents> distance;
  if ( distName == "KullbackLeibler" )
    distance = boost::shared_ptr<TSOSDistanceBetweenComponents>(new TSOSKullbackLeiblerDistance());
  else if ( distName == "Mahalanobis" )
    distance = boost::shared_ptr<TSOSDistanceBetweenComponents>(new TSOSMahalanobisDistance());
  
  return distance;
}


