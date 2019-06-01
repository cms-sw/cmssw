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

GsfMaterialEffectsESProducer::GsfMaterialEffectsESProducer(const edm::ParameterSet& p) {
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this, myname);
}

GsfMaterialEffectsESProducer::~GsfMaterialEffectsESProducer() {}

std::unique_ptr<GsfMaterialEffectsUpdator> GsfMaterialEffectsESProducer::produce(
    const TrackingComponentsRecord& iRecord) {
  double mass = pset_.getParameter<double>("Mass");
  std::string msName = pset_.getParameter<std::string>("MultipleScatteringUpdator");
  std::string elName = pset_.getParameter<std::string>("EnergyLossUpdator");

  std::unique_ptr<GsfMaterialEffectsUpdator> msUpdator;
  if (msName == "GsfMultipleScatteringUpdator") {
    msUpdator.reset(new GsfMultipleScatteringUpdator(mass));
  } else {
    msUpdator.reset(new GsfMaterialEffectsAdapter(MultipleScatteringUpdator(mass)));
  }

  std::unique_ptr<GsfMaterialEffectsUpdator> elUpdator;
  if (elName == "GsfBetheHeitlerUpdator") {
    std::string fileName = pset_.getParameter<std::string>("BetheHeitlerParametrization");
    int correction = pset_.getParameter<int>("BetheHeitlerCorrection");
    elUpdator.reset(new GsfBetheHeitlerUpdator(fileName, correction));
  } else {
    elUpdator.reset(new GsfMaterialEffectsAdapter(EnergyLossUpdator(mass)));
  }

  auto updator = std::make_unique<GsfCombinedMaterialEffectsUpdator>(*msUpdator, *elUpdator);

  return updator;
}
