#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimTracker/TrackAssociation/interface/CosmicParametersDefinerForTP.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include <DataFormats/GeometrySurface/interface/GloballyPositioned.h>
#include <DataFormats/GeometrySurface/interface/Surface.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>

class TrajectoryStateClosestToBeamLineBuilder;

CosmicParametersDefinerForTP::CosmicParametersDefinerForTP(edm::ConsumesCollector iC)
    : ParametersDefinerForTP(edm::InputTag("offlineBeamSpot"), iC), geometryToken_(iC.esConsumes()) {}
CosmicParametersDefinerForTP::~CosmicParametersDefinerForTP() = default;

TrackingParticle::Vector CosmicParametersDefinerForTP::momentum(const edm::Event &iEvent,
                                                                const edm::EventSetup &iSetup,
                                                                const TrackingParticleRef &tpr) const {
  // to add a new implementation for cosmic. For the moment, it is just as for
  // the base class:
  using namespace edm;
  using namespace std;
  using namespace reco;

  auto const &bs = iEvent.get(bsToken_);
  auto const &geometry = iSetup.getData(geometryToken_);
  auto const &mf = iSetup.getData(mfToken_);

  GlobalVector finalGV(0, 0, 0);
  GlobalPoint finalGP(0, 0, 0);
  double radius(9999);
  bool found(false);
  TrackingParticle::Vector momentum(0, 0, 0);

  edm::LogVerbatim("CosmicParametersDefinerForTP") << "\t in CosmicParametersDefinerForTP::momentum";
  edm::LogVerbatim("CosmicParametersDefinerForTP")
      << "\t \t Original TP state:          pt = " << tpr->pt() << ", pz = " << tpr->pz();

  if (simHitsTPAssoc.isValid() == 0) {
    LogError("TrackAssociation") << "Invalid handle!";
    return momentum;
  }
  std::pair<TrackingParticleRef, TrackPSimHitRef> clusterTPpairWithDummyTP(
      tpr,
      TrackPSimHitRef());  // SimHit is dummy: for simHitTPAssociationListGreater
                           // sorting only the cluster is needed
  auto range = std::equal_range(simHitsTPAssoc->begin(),
                                simHitsTPAssoc->end(),
                                clusterTPpairWithDummyTP,
                                SimHitTPAssociationProducer::simHitTPAssociationListGreater);
  for (auto ip = range.first; ip != range.second; ++ip) {
    TrackPSimHitRef it = ip->second;
    const GeomDet *tmpDet = geometry.idToDet(DetId(it->detUnitId()));
    if (!tmpDet) {
      edm::LogVerbatim("CosmicParametersDefinerForTP")
          << "***WARNING in CosmicParametersDefinerForTP::momentum: no GeomDet "
             "for: "
          << it->detUnitId() << ". Skipping it."
          << "\n";
      continue;
    }

    LocalVector lv = it->momentumAtEntry();
    Local3DPoint lp = it->localPosition();
    GlobalVector gv = tmpDet->surface().toGlobal(lv);
    GlobalPoint gp = tmpDet->surface().toGlobal(lp);

    // discard hits related to low energy debris from the primary particle
    if (it->processType() != 0)
      continue;

    if (gp.perp() < radius) {
      found = true;
      radius = gp.perp();
      finalGV = gv;
      finalGP = gp;
    }
  }

  edm::LogVerbatim("CosmicParametersDefinerForTP")
      //   <<"\t FINAL State at InnerMost Hit: Radius = "<< finalGP.perp() << ",
      //   z = "<< finalGP.z()
      //  <<", pt = "<< finalGV.perp() << ", pz = "<< finalGV.z();
      << "\t \t FINAL State at InnerMost Hit:   pt = " << finalGV.perp() << ", pz = " << finalGV.z();

  if (found) {
    FreeTrajectoryState ftsAtProduction(finalGP, finalGV, TrackCharge(tpr->charge()), &mf);
    TSCBLBuilderNoMaterial tscblBuilder;
    TrajectoryStateClosestToBeamLine tsAtClosestApproach =
        tscblBuilder(ftsAtProduction, bs);  // as in TrackProducerAlgorithm

    if (tsAtClosestApproach.isValid()) {
      GlobalVector p = tsAtClosestApproach.trackStateAtPCA().momentum();
      momentum = TrackingParticle::Vector(p.x(), p.y(), p.z());
    } else {
      edm::LogVerbatim("CosmicParametersDefinerForTP") << "*** WARNING in CosmicParametersDefinerForTP::momentum: "
                                                          "tsAtClosestApproach is not valid."
                                                       << "\n";
    }

    edm::LogVerbatim("CosmicParametersDefinerForTP")
        << "\t \t FINAL State extrap. at PCA: pt = " << sqrt(momentum.x() * momentum.x() + momentum.y() * momentum.y())
        << ", pz = " << momentum.z() << "\n";

    return momentum;
  }

  edm::LogVerbatim("CosmicParametersDefinerForTP")
      << "*** WARNING in CosmicParametersDefinerForTP::momentum: NOT found the "
         "innermost TP point"
      << "\n";
  edm::LogVerbatim("CosmicParametersDefinerForTP")
      << "*** FINAL Reference MOMENTUM TP (px,py,pz) = " << momentum.x() << momentum.y() << momentum.z() << "\n";
  return momentum;
}

TrackingParticle::Point CosmicParametersDefinerForTP::vertex(const edm::Event &iEvent,
                                                             const edm::EventSetup &iSetup,
                                                             const TrackingParticleRef &tpr) const {
  using namespace edm;
  using namespace std;
  using namespace reco;

  auto const &bs = iEvent.get(bsToken_);
  auto const &geometry = iSetup.getData(geometryToken_);
  auto const &mf = iSetup.getData(mfToken_);

  GlobalVector finalGV(0, 0, 0);
  GlobalPoint finalGP(0, 0, 0);
  double radius(9999);
  bool found(false);
  TrackingParticle::Point vertex(0, 0, 0);

  edm::LogVerbatim("CosmicParametersDefinerForTP") << "\t in CosmicParametersDefinerForTP::vertex";
  edm::LogVerbatim("CosmicParametersDefinerForTP")
      << "\t \t Original TP state:          radius = "
      << sqrt(tpr->vertex().x() * tpr->vertex().x() + tpr->vertex().y() * tpr->vertex().y())
      << ", z = " << tpr->vertex().z();

  if (simHitsTPAssoc.isValid() == 0) {
    LogError("TrackAssociation") << "Invalid handle!";
    return vertex;
  }
  std::pair<TrackingParticleRef, TrackPSimHitRef> clusterTPpairWithDummyTP(
      tpr,
      TrackPSimHitRef());  // SimHit is dummy: for simHitTPAssociationListGreater
                           // sorting only the cluster is needed
  auto range = std::equal_range(simHitsTPAssoc->begin(),
                                simHitsTPAssoc->end(),
                                clusterTPpairWithDummyTP,
                                SimHitTPAssociationProducer::simHitTPAssociationListGreater);
  for (auto ip = range.first; ip != range.second; ++ip) {
    TrackPSimHitRef it = ip->second;
    const GeomDet *tmpDet = geometry.idToDet(DetId(it->detUnitId()));
    if (!tmpDet) {
      edm::LogVerbatim("CosmicParametersDefinerForTP")
          << "***WARNING in CosmicParametersDefinerForTP::vertex: no GeomDet "
             "for: "
          << it->detUnitId() << ". Skipping it."
          << "\n";
      continue;
    }

    LocalVector lv = it->momentumAtEntry();
    Local3DPoint lp = it->localPosition();
    GlobalVector gv = tmpDet->surface().toGlobal(lv);
    GlobalPoint gp = tmpDet->surface().toGlobal(lp);

    // discard hits related to low energy debris from the primary particle
    if (it->processType() != 0)
      continue;

    if (gp.perp() < radius) {
      found = true;
      radius = gp.perp();
      finalGV = gv;
      finalGP = gp;
    }
  }
  edm::LogVerbatim("CosmicParametersDefinerForTP")
      << "\t \t FINAL State at InnerMost Hit:   radius = " << finalGP.perp() << ", z = " << finalGP.z();

  if (found) {
    FreeTrajectoryState ftsAtProduction(finalGP, finalGV, TrackCharge(tpr->charge()), &mf);
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
      edm::LogVerbatim("CosmicParametersDefinerForTP") << "*** WARNING in CosmicParametersDefinerForTP::vertex: "
                                                          "tsAtClosestApproach is not valid."
                                                       << "\n";
    }
    edm::LogVerbatim("CosmicParametersDefinerForTP")
        << "\t \t FINAL State extrap. at PCA: radius = " << sqrt(vertex.x() * vertex.x() + vertex.y() * vertex.y())
        << ", z = " << vertex.z() << "\n";

    return vertex;
  }

  edm::LogVerbatim("CosmicParametersDefinerForTP")
      << "*** WARNING in CosmicParametersDefinerForTP::vertex: NOT found the "
         "innermost TP point"
      << "\n";
  edm::LogVerbatim("CosmicParametersDefinerForTP")
      << "*** FINAL Reference VERTEX TP   V(x,y,z) = " << vertex.x() << vertex.y() << vertex.z() << "\n";

  return vertex;
}
