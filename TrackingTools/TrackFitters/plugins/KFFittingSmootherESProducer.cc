

// to be included only here...
#include "KFFittingSmoother.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitterRecord.h"

namespace {

  class KFFittingSmootherESProducer final : public edm::ESProducer {
  public:
    KFFittingSmootherESProducer(const edm::ParameterSet& p) : pset_{p} {
      std::string myname = p.getParameter<std::string>("ComponentName");
      auto cc = setWhatProduced(this, myname);
      fitToken_ = cc.consumes(edm::ESInputTag("", pset_.getParameter<std::string>("Fitter")));
      smoothToken_ = cc.consumes(edm::ESInputTag("", pset_.getParameter<std::string>("Smoother")));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("ComponentName", "KFFittingSmoother");
      desc.add<std::string>("Fitter", "KFFitter");
      desc.add<std::string>("Smoother", "KFSmoother");
      KFFittingSmoother::fillDescriptions(desc);
      descriptions.add("KFFittingSmoother", desc);
    }

    std::unique_ptr<TrajectoryFitter> produce(const TrajectoryFitterRecord& iRecord) {
      return std::make_unique<KFFittingSmoother>(iRecord.get(fitToken_), iRecord.get(smoothToken_), pset_);
    }

  private:
    const edm::ParameterSet pset_;
    edm::ESGetToken<TrajectoryFitter, TrajectoryFitterRecord> fitToken_;
    edm::ESGetToken<TrajectorySmoother, TrajectoryFitterRecord> smoothToken_;
  };
}  // namespace

#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_EVENTSETUP_MODULE(KFFittingSmootherESProducer);
