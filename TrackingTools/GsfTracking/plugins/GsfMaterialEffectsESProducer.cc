#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"

#include "TrackingTools/MaterialEffects/interface/MultipleScatteringUpdator.h"
#include "TrackingTools/MaterialEffects/interface/EnergyLossUpdator.h"
#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsAdapter.h"
#include "TrackingTools/GsfTracking/interface/GsfMultipleScatteringUpdator.h"
#include "TrackingTools/GsfTracking/interface/GsfBetheHeitlerUpdator.h"
#include "TrackingTools/GsfTracking/interface/GsfCombinedMaterialEffectsUpdator.h"

#include <memory>

#include <string>
#include <memory>
#include <optional>

/** Provides algorithms for estimating material effects (GSF compatible).
 * Multiple scattering estimates can be provided according to a single (== "KF") 
 * or two-component model. Energy loss estimates can be provided according to 
 * a single component ionization- or radiation model (== "KF") or a multi-component
 * Bethe-Heitler model. */

class GsfMaterialEffectsESProducer : public edm::ESProducer {
public:
  GsfMaterialEffectsESProducer(const edm::ParameterSet& p);

  std::unique_ptr<GsfMaterialEffectsUpdator> produce(const TrackingComponentsRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  struct BetheHeitlerInit {
    BetheHeitlerInit(std::string iName, int iCorrection) : fileName(std::move(iName)), correction(iCorrection) {}
    std::string fileName;
    int correction;
  };

  std::optional<BetheHeitlerInit> doInit(const edm::ParameterSet& p) {
    if (p.getParameter<std::string>("EnergyLossUpdator") != "GsfBetheHeitlerUpdator") {
      return std::optional<BetheHeitlerInit>();
    }
    return std::make_optional<BetheHeitlerInit>(p.getParameter<std::string>("BetheHeitlerParametrization"),
                                                p.getParameter<int>("BetheHeitlerCorrection"));
  }

  std::optional<BetheHeitlerInit> betheHeitlerInit_;
  const double mass_;
  const bool useMultipleScattering_;
};

using namespace edm;

GsfMaterialEffectsESProducer::GsfMaterialEffectsESProducer(const edm::ParameterSet& p)
    : betheHeitlerInit_(doInit(p)),
      mass_(p.getParameter<double>("Mass")),
      useMultipleScattering_(p.getParameter<std::string>("MultipleScatteringUpdator") == "GsfMultipleScatteringUpdator")

{
  std::string myname = p.getParameter<std::string>("ComponentName");
  setWhatProduced(this, myname);
}

std::unique_ptr<GsfMaterialEffectsUpdator> GsfMaterialEffectsESProducer::produce(
    const TrackingComponentsRecord& iRecord) {
  std::unique_ptr<GsfMaterialEffectsUpdator> msUpdator;
  if (useMultipleScattering_) {
    msUpdator = std::make_unique<GsfMultipleScatteringUpdator>(mass_);
  } else {
    msUpdator = std::make_unique<GsfMaterialEffectsAdapter>(MultipleScatteringUpdator(mass_));
  }

  std::unique_ptr<GsfMaterialEffectsUpdator> elUpdator;
  if (betheHeitlerInit_) {
    elUpdator = std::make_unique<GsfBetheHeitlerUpdator>(betheHeitlerInit_->fileName, betheHeitlerInit_->correction);
  } else {
    elUpdator = std::make_unique<GsfMaterialEffectsAdapter>(EnergyLossUpdator(mass_));
  }

  auto updator = std::make_unique<GsfCombinedMaterialEffectsUpdator>(*msUpdator, *elUpdator);

  return updator;
}

void GsfMaterialEffectsESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("ComponentName");
  desc.add<double>("Mass");
  desc.add<std::string>("MultipleScatteringUpdator");
  //Depending on the value of "EnergyLossUpdator", different parameters are allowed
  desc.ifValue(
      edm::ParameterDescription<std::string>("EnergyLossUpdator", "GsfBetheHeitlerUpdator", true),
      "GsfBetheHeitlerUpdator" >> (edm::ParameterDescription<std::string>("BetheHeitlerParametrization", true) and
                                   edm::ParameterDescription<int>("BetheHeitlerCorrection", true)) or
          "EnergyLossUpdator" >> edm::EmptyGroupDescription()  //No additional parameters needed
  );
  descriptions.addDefault(desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(GsfMaterialEffectsESProducer);
