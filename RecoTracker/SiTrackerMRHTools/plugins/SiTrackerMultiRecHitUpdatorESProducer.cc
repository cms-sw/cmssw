// user includes
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/Record/interface/MultiRecHitRecord.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdator.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

// system includes
#include <string>
#include <memory>

class SiTrackerMultiRecHitUpdatorESProducer : public edm::ESProducer {
public:
  SiTrackerMultiRecHitUpdatorESProducer(const edm::ParameterSet& p);
  ~SiTrackerMultiRecHitUpdatorESProducer() override = default;
  std::unique_ptr<SiTrackerMultiRecHitUpdator> produce(const MultiRecHitRecord&);

private:
  std::string myname_, sname_, hitpropagator_;
  bool debug_;
  std::vector<double> annealingProgram_;
  float chi2Cut1D_, chi2Cut2D_;

  edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> ttrhbToken;
  edm::ESGetToken<TrackingRecHitPropagator, CkfComponentsRecord> ttrhpToken;
};

using namespace edm;

SiTrackerMultiRecHitUpdatorESProducer::SiTrackerMultiRecHitUpdatorESProducer(const edm::ParameterSet& p)
    : myname_(p.getParameter<std::string>("ComponentName")),
      sname_(p.getParameter<std::string>("TTRHBuilder")),
      hitpropagator_(p.getParameter<std::string>("HitPropagator")),
      debug_(p.getParameter<bool>("Debug")),
      annealingProgram_(p.getParameter<std::vector<double> >("AnnealingProgram")),
      chi2Cut1D_(p.getParameter<double>("ChiSquareCut1D")),
      chi2Cut2D_(p.getParameter<double>("ChiSquareCut2D")) {
  auto cc = setWhatProduced(this, myname_);
  ttrhbToken = cc.consumes(edm::ESInputTag("", sname_));
  ttrhpToken = cc.consumes(edm::ESInputTag("", hitpropagator_));
}

std::unique_ptr<SiTrackerMultiRecHitUpdator> SiTrackerMultiRecHitUpdatorESProducer::produce(
    const MultiRecHitRecord& iRecord) {
  const TransientTrackingRecHitBuilder* hbuilder = &iRecord.get(ttrhbToken);
  const TrackingRecHitPropagator* hhitpropagator = &iRecord.get(ttrhpToken);

  return std::make_unique<SiTrackerMultiRecHitUpdator>(
      hbuilder, hhitpropagator, chi2Cut1D_, chi2Cut2D_, annealingProgram_, debug_);
}

DEFINE_FWK_EVENTSETUP_MODULE(SiTrackerMultiRecHitUpdatorESProducer);
