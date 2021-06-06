#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitterRecord.h"
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"
#include "TrackingTools/GsfTracking/interface/GsfPropagatorWithMaterial.h"
#include "TrackingTools/GsfTracking/interface/GsfMultiStateUpdator.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianStateMerger.h"
#include "TrackingTools/GsfTools/interface/CloseComponentsMerger.h"
#include "TrackingTools/GsfTracking/interface/MultiTrajectoryStateMerger.h"
#include "TrackingTools/GsfTracking/interface/GsfChi2MeasurementEstimator.h"
#include "TrackingTools/GsfTracking/interface/GsfTrajectorySmoother.h"

#include <string>
#include <memory>

/** Provides a GSF smoother algorithm */

class GsfTrajectorySmootherESProducer : public edm::ESProducer {
public:
  GsfTrajectorySmootherESProducer(const edm::ParameterSet& p);

  std::unique_ptr<TrajectorySmoother> produce(const TrajectoryFitterRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::ESGetToken<GsfMaterialEffectsUpdator, TrackingComponentsRecord> matToken_;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorToken_;
  edm::ESGetToken<MultiGaussianStateMerger<5>, TrackingComponentsRecord> mergerToken_;
  edm::ESGetToken<DetLayerGeometry, RecoGeometryRecord> geoToken_;
  const double scale_;
};

GsfTrajectorySmootherESProducer::GsfTrajectorySmootherESProducer(const edm::ParameterSet& p)
    : scale_(p.getParameter<double>("ErrorRescaling")) {
  std::string myname = p.getParameter<std::string>("ComponentName");
  auto cc = setWhatProduced(this, myname);
  matToken_ = cc.consumes(edm::ESInputTag("", p.getParameter<std::string>("MaterialEffectsUpdator")));
  propagatorToken_ = cc.consumes(edm::ESInputTag("", p.getParameter<std::string>("GeometricalPropagator")));
  mergerToken_ = cc.consumes(edm::ESInputTag("", p.getParameter<std::string>("Merger")));
  geoToken_ = cc.consumes(edm::ESInputTag("", p.getParameter<std::string>("RecoGeometry")));
}

std::unique_ptr<TrajectorySmoother> GsfTrajectorySmootherESProducer::produce(const TrajectoryFitterRecord& iRecord) {
  //
  // propagator
  //
  GsfPropagatorWithMaterial propagator(iRecord.get(propagatorToken_), iRecord.get(matToken_));
  //
  // merger
  //
  MultiTrajectoryStateMerger merger(iRecord.get(mergerToken_));
  //
  // estimator
  //
  //   double chi2Cut = pset_.getParameter<double>("ChiSquarCut");
  double chi2Cut(100.);
  GsfChi2MeasurementEstimator estimator(chi2Cut);
  //
  // geometry
  // create algorithm
  //
  //   bool matBefUpd = pset_.getParameter<bool>("MaterialBeforeUpdate");
  return std::make_unique<GsfTrajectorySmoother>(
      propagator,
      GsfMultiStateUpdator(),
      estimator,
      merger,
      // 									 matBefUpd,
      scale_,
      true,  //BM should this be taken from parameterSet?
      &iRecord.get(geoToken_));
}

void GsfTrajectorySmootherESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName");
  desc.add<std::string>("MaterialEffectsUpdator");
  desc.add<std::string>("GeometricalPropagator");
  desc.add<std::string>("Merger");
  desc.add<std::string>("RecoGeometry");
  desc.add<double>("ErrorRescaling");

  descriptions.addDefault(desc);
}
DEFINE_FWK_EVENTSETUP_MODULE(GsfTrajectorySmootherESProducer);
