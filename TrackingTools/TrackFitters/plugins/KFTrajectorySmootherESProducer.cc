#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryFitterRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include <memory>

namespace {

  class KFTrajectorySmootherESProducer final : public edm::ESProducer {
  public:
    KFTrajectorySmootherESProducer(const edm::ParameterSet& p);
    std::unique_ptr<TrajectorySmoother> produce(const TrajectoryFitterRecord&);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("ComponentName", "KFSmoother");
      desc.add<std::string>("Propagator", "PropagatorWithMaterial");
      desc.add<std::string>("Updator", "KFUpdator");
      desc.add<std::string>("Estimator", "Chi2");
      desc.add<std::string>("RecoGeometry", "GlobalDetLayerGeometry");
      desc.add<double>("errorRescaling", 100);
      desc.add<int>("minHits", 3);
      descriptions.add("KFTrajectorySmoother", desc);
    }

  private:
    edm::ESGetToken<Propagator, TrackingComponentsRecord> propToken_;
    edm::ESGetToken<TrajectoryStateUpdator, TrackingComponentsRecord> updToken_;
    edm::ESGetToken<Chi2MeasurementEstimatorBase, TrackingComponentsRecord> estToken_;
    edm::ESGetToken<DetLayerGeometry, RecoGeometryRecord> geoToken_;
    const double rescaleFactor_;
    const int minHits_;
  };

  KFTrajectorySmootherESProducer::KFTrajectorySmootherESProducer(const edm::ParameterSet& p)
      : rescaleFactor_{p.getParameter<double>("errorRescaling")}, minHits_{p.getParameter<int>("minHits")} {
    std::string myname = p.getParameter<std::string>("ComponentName");
    auto cc = setWhatProduced(this, myname);
    propToken_ = cc.consumes(edm::ESInputTag("", p.getParameter<std::string>("Propagator")));
    updToken_ = cc.consumes(edm::ESInputTag("", p.getParameter<std::string>("Updator")));
    estToken_ = cc.consumes(edm::ESInputTag("", p.getParameter<std::string>("Estimator")));
    geoToken_ = cc.consumes(edm::ESInputTag("", p.getParameter<std::string>("RecoGeometry")));
  }

  std::unique_ptr<TrajectorySmoother> KFTrajectorySmootherESProducer::produce(const TrajectoryFitterRecord& iRecord) {
    return std::make_unique<KFTrajectorySmoother>(&iRecord.get(propToken_),
                                                  &iRecord.get(updToken_),
                                                  &iRecord.get(estToken_),
                                                  rescaleFactor_,
                                                  minHits_,
                                                  &iRecord.get(geoToken_));
  }
}  // namespace

#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(KFTrajectorySmootherESProducer);
