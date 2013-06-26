#include "TrackingTools/GsfTracking/plugins/GsfMaterialEffectsESProducer.h"

#include "TrackingTools/MaterialEffects/interface/MultipleScatteringUpdator.h"
#include "TrackingTools/MaterialEffects/interface/EnergyLossUpdator.h"
#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsAdapter.h"
#include "TrackingTools/GsfTracking/interface/GsfMultipleScatteringUpdator.h"
#include "TrackingTools/GsfTracking/interface/GsfBetheHeitlerUpdator.h"
#include "TrackingTools/GsfTracking/interface/GsfCombinedMaterialEffectsUpdator.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;

GsfMaterialEffectsESProducer::GsfMaterialEffectsESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

GsfMaterialEffectsESProducer::~GsfMaterialEffectsESProducer() {}

boost::shared_ptr<GsfMaterialEffectsUpdator> 
GsfMaterialEffectsESProducer::produce(const TrackingComponentsRecord & iRecord){ 
  double mass = pset_.getParameter<double>("Mass");
  std::string msName = pset_.getParameter<std::string>("MultipleScatteringUpdator");
  std::string elName = pset_.getParameter<std::string>("EnergyLossUpdator");

  GsfMaterialEffectsUpdator* msUpdator;
  if ( msName == "GsfMultipleScatteringUpdator" ) {
    msUpdator = new GsfMultipleScatteringUpdator(mass);
  }
  else {
    msUpdator = new GsfMaterialEffectsAdapter(MultipleScatteringUpdator(mass));
  }
  
  GsfMaterialEffectsUpdator* elUpdator;
  if ( elName == "GsfBetheHeitlerUpdator" ) {
    std::string fileName = pset_.getParameter<std::string>("BetheHeitlerParametrization");
    int correction = pset_.getParameter<int>("BetheHeitlerCorrection");
    elUpdator = new GsfBetheHeitlerUpdator(fileName,correction);
  }
  else {
    elUpdator = new GsfMaterialEffectsAdapter(EnergyLossUpdator(mass));
  }

  boost::shared_ptr<GsfMaterialEffectsUpdator> updator =
    boost::shared_ptr<GsfMaterialEffectsUpdator>(new GsfCombinedMaterialEffectsUpdator(*msUpdator,
										       *elUpdator));
  delete msUpdator;
  delete elUpdator;

  return updator;
}


