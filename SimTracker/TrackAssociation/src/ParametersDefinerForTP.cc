#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "FWCore/Framework/interface/Event.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimTracker/TrackAssociation/interface/ParametersDefinerForTP.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
class TrajectoryStateClosestToBeamLineBuilder;

ParametersDefinerForTP::ParametersDefinerForTP(const edm::InputTag &beamspot, edm::ConsumesCollector iC)
    : bsToken_(iC.consumes(beamspot)), mfToken_(iC.esConsumes()) {}

ParametersDefinerForTP::~ParametersDefinerForTP() = default;

TrackingParticle::Vector ParametersDefinerForTP::momentum(const edm::Event &iEvent,
                                                          const edm::EventSetup &iSetup,
                                                          const Charge charge,
                                                          const Point &vtx,
                                                          const LorentzVector &lv) const {
  // to add a new implementation for cosmic. For the moment, it is just as for
  // the base class:

  using namespace edm;

  auto const &bs = iEvent.get(bsToken_);
  auto const &mf = iSetup.getData(mfToken_);

  TrackingParticle::Vector momentum(0, 0, 0);

  FreeTrajectoryState ftsAtProduction(
      GlobalPoint(vtx.x(), vtx.y(), vtx.z()), GlobalVector(lv.x(), lv.y(), lv.z()), TrackCharge(charge), &mf);

  TSCBLBuilderNoMaterial tscblBuilder;
  TrajectoryStateClosestToBeamLine tsAtClosestApproach =
      tscblBuilder(ftsAtProduction, bs);  // as in TrackProducerAlgorithm
  if (tsAtClosestApproach.isValid()) {
    GlobalVector p = tsAtClosestApproach.trackStateAtPCA().momentum();
    momentum = TrackingParticle::Vector(p.x(), p.y(), p.z());
  }
  return momentum;
}

TrackingParticle::Point ParametersDefinerForTP::vertex(const edm::Event &iEvent,
                                                       const edm::EventSetup &iSetup,
                                                       const Charge charge,
                                                       const Point &vtx,
                                                       const LorentzVector &lv) const {
  // to add a new implementation for cosmic. For the moment, it is just as for
  // the base class:
  using namespace edm;

  auto const &bs = iEvent.get(bsToken_);
  auto const &mf = iSetup.getData(mfToken_);

  TrackingParticle::Point vertex(0, 0, 0);

  FreeTrajectoryState ftsAtProduction(
      GlobalPoint(vtx.x(), vtx.y(), vtx.z()), GlobalVector(lv.x(), lv.y(), lv.z()), TrackCharge(charge), &mf);

  TSCBLBuilderNoMaterial tscblBuilder;
  TrajectoryStateClosestToBeamLine tsAtClosestApproach =
      tscblBuilder(ftsAtProduction, bs);  // as in TrackProducerAlgorithm
  if (tsAtClosestApproach.isValid()) {
    GlobalPoint v = tsAtClosestApproach.trackStateAtPCA().position();
    vertex = TrackingParticle::Point(v.x(), v.y(), v.z());
  } else {
    // to preserve old behaviour
    // would be better to flag this somehow to allow ignoring in downstream
    vertex = TrackingParticle::Point(bs.x0(), bs.y0(), bs.z0());
  }
  return vertex;
}

std::tuple<TrackingParticle::Vector, TrackingParticle::Point> ParametersDefinerForTP::momentumAndVertex(
    const edm::Event &iEvent,
    const edm::EventSetup &iSetup,
    const Charge charge,
    const Point &vtx,
    const LorentzVector &lv) const {
  using namespace edm;

  auto const &bs = iEvent.get(bsToken_);
  auto const &mf = iSetup.getData(mfToken_);

  TrackingParticle::Point vertex(bs.x0(), bs.y0(), bs.z0());
  TrackingParticle::Vector momentum(0, 0, 0);

  FreeTrajectoryState ftsAtProduction(
      GlobalPoint(vtx.x(), vtx.y(), vtx.z()), GlobalVector(lv.x(), lv.y(), lv.z()), TrackCharge(charge), &mf);

  TSCBLBuilderNoMaterial tscblBuilder;
  TrajectoryStateClosestToBeamLine tsAtClosestApproach =
      tscblBuilder(ftsAtProduction, bs);  // as in TrackProducerAlgorithm
  if (tsAtClosestApproach.isValid()) {
    GlobalPoint v = tsAtClosestApproach.trackStateAtPCA().position();
    vertex = TrackingParticle::Point(v.x(), v.y(), v.z());
    GlobalVector p = tsAtClosestApproach.trackStateAtPCA().momentum();
    momentum = TrackingParticle::Vector(p.x(), p.y(), p.z());
    ;
  }

  return std::make_tuple(momentum, vertex);
}
