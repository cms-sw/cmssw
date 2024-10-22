#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitterRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"
#include "TrackingTools/GsfTracking/interface/GsfPropagatorWithMaterial.h"
#include "TrackingTools/GsfTracking/interface/GsfMultiStateUpdator.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianStateMerger.h"
#include "TrackingTools/GsfTools/interface/CloseComponentsMerger.h"
#include "TrackingTools/GsfTracking/interface/MultiTrajectoryStateMerger.h"
#include "TrackingTools/GsfTracking/interface/GsfChi2MeasurementEstimator.h"
#include "TrackingTools/GsfTracking/interface/GsfTrajectoryFitter.h"

#include <string>
#include <memory>

#include <iostream>

/** Provides a GSF fitter algorithm */

class GsfTrajectoryFitterESProducer : public edm::ESProducer {
public:
  GsfTrajectoryFitterESProducer(const edm::ParameterSet& p);
  ~GsfTrajectoryFitterESProducer() override;
  std::unique_ptr<TrajectoryFitter> produce(const TrajectoryFitterRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::ESGetToken<GsfMaterialEffectsUpdator, TrackingComponentsRecord> matUpdatorToken_;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorToken_;
  edm::ESGetToken<MultiGaussianStateMerger<5>, TrackingComponentsRecord> mergerToken_;
  edm::ESGetToken<DetLayerGeometry, RecoGeometryRecord> geoToken_;
};

GsfTrajectoryFitterESProducer::GsfTrajectoryFitterESProducer(const edm::ParameterSet& p) {
  std::string myname = p.getParameter<std::string>("ComponentName");
  auto cc = setWhatProduced(this, myname);
  matUpdatorToken_ = cc.consumes(edm::ESInputTag("", p.getParameter<std::string>("MaterialEffectsUpdator")));
  propagatorToken_ = cc.consumes(edm::ESInputTag("", p.getParameter<std::string>("GeometricalPropagator")));
  mergerToken_ = cc.consumes(edm::ESInputTag("", p.getParameter<std::string>("Merger")));
  geoToken_ = cc.consumes(edm::ESInputTag("", p.getParameter<std::string>("RecoGeometry")));
}

GsfTrajectoryFitterESProducer::~GsfTrajectoryFitterESProducer() {}

std::unique_ptr<TrajectoryFitter> GsfTrajectoryFitterESProducer::produce(const TrajectoryFitterRecord& iRecord) {
  //
  // propagator
  //
  GsfPropagatorWithMaterial propagator(iRecord.get(propagatorToken_), iRecord.get(matUpdatorToken_));
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
  // create algorithm
  //
  return std::make_unique<GsfTrajectoryFitter>(
      propagator, GsfMultiStateUpdator(), estimator, merger, &iRecord.get(geoToken_));
}

void GsfTrajectoryFitterESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName");
  desc.add<std::string>("MaterialEffectsUpdator");
  desc.add<std::string>("GeometricalPropagator");
  desc.add<std::string>("Merger");
  desc.add<std::string>("RecoGeometry");
  descriptions.addDefault(desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(GsfTrajectoryFitterESProducer);
