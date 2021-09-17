#include "PropagatorWithMaterialESProducer.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/ESInputTag.h"

#include <string>
#include <memory>

using namespace edm;

namespace {
  PropagationDirection stringToDirection(std::string const& iName) {
    PropagationDirection dir = alongMomentum;

    if (iName == "oppositeToMomentum")
      dir = oppositeToMomentum;
    if (iName == "alongMomentum")
      dir = alongMomentum;
    if (iName == "anyDirection")
      dir = anyDirection;
    return dir;
  }
}  // namespace

PropagatorWithMaterialESProducer::PropagatorWithMaterialESProducer(const edm::ParameterSet& p)
    : mfToken_(setWhatProduced(this, p.getParameter<std::string>("ComponentName"))
                   .consumesFrom<MagneticField, IdealMagneticFieldRecord>(
                       edm::ESInputTag("", p.getParameter<std::string>("SimpleMagneticField")))),
      mass_(p.getParameter<double>("Mass")),
      maxDPhi_(p.getParameter<double>("MaxDPhi")),
      ptMin_(p.getParameter<double>("ptMin")),
      dir_(stringToDirection(p.getParameter<std::string>("PropagationDirection"))),
      useRK_(p.getParameter<bool>("useRungeKutta")),
      useOldAnalPropLogic_(p.getParameter<bool>("useOldAnalPropLogic")) {}

std::unique_ptr<Propagator> PropagatorWithMaterialESProducer::produce(const TrackingComponentsRecord& iRecord) {
  //  edm::ESInputTag mfESInputTag(mfName);
  //  iRecord.getRecord<IdealMagneticFieldRecord>().get(mfESInputTag,magfield);
  //fixme check that useRK is false when using SimpleMagneticField

  return std::make_unique<PropagatorWithMaterial>(
      dir_, mass_, &iRecord.get(mfToken_), maxDPhi_, useRK_, ptMin_, useOldAnalPropLogic_);
}

void PropagatorWithMaterialESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  ParameterSetDescription desc;
  desc.add<std::string>("PropagationDirection");
  desc.add<std::string>("SimpleMagneticField", "");
  desc.add<std::string>("ComponentName");
  desc.add<double>("Mass");
  desc.add<double>("MaxDPhi");
  desc.add<bool>("useRungeKutta");
  desc.add<bool>("useOldAnalPropLogic", true);
  desc.add<double>("ptMin", -1.0);
  descriptions.addDefault(desc);
}
