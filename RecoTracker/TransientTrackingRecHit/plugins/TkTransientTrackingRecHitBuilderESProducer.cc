#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoLocalTracker/Records/interface/TrackerCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include <string>
#include <memory>
#include <optional>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class TkTransientTrackingRecHitBuilderESProducer : public edm::ESProducer {
public:
  TkTransientTrackingRecHitBuilderESProducer(const edm::ParameterSet& p);

  std::unique_ptr<TransientTrackingRecHitBuilder> produce(const TransientRecHitRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::optional<edm::ESGetToken<StripClusterParameterEstimator, TkStripCPERecord>> spToken_;
  std::optional<edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord>> ppToken_;
  std::optional<edm::ESGetToken<SiStripRecHitMatcher, TkStripCPERecord>> mpToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  std::optional<edm::ESGetToken<ClusterParameterEstimator<Phase2TrackerCluster1D>, TkPhase2OTCPERecord>> p2OTToken_;
  bool const computeCoarseLocalPositionFromDisk_;
};

namespace {
  template <typename T, typename R>
  void setConsumes(edm::ESConsumesCollector& iCollector,
                   std::optional<edm::ESGetToken<T, R>>& iToken,
                   std::string const& iLabel) {
    iToken = iCollector.consumesFrom<T, R>(edm::ESInputTag("", iLabel));
  }
}  // namespace

using namespace edm;

TkTransientTrackingRecHitBuilderESProducer::TkTransientTrackingRecHitBuilderESProducer(const edm::ParameterSet& p)
    : computeCoarseLocalPositionFromDisk_(p.getParameter<bool>("ComputeCoarseLocalPositionFromDisk")) {
  std::string const myname = p.getParameter<std::string>("ComponentName");
  auto c = setWhatProduced(this, myname);
  c.setConsumes(geomToken_);

  std::string const sname = p.getParameter<std::string>("StripCPE");
  if (sname != "Fake") {
    setConsumes(c, spToken_, sname);
  }

  std::string const pname = p.getParameter<std::string>("PixelCPE");
  if (pname != "Fake") {
    setConsumes(c, ppToken_, pname);
  }

  auto const mname = p.getParameter<std::string>("Matcher");
  if (mname != "Fake") {
    setConsumes(c, mpToken_, mname);
  }

  auto const P2otname = p.getParameter<std::string>("Phase2StripCPE");
  if (!P2otname.empty()) {
    setConsumes(c, p2OTToken_, P2otname);
  }
}

std::unique_ptr<TransientTrackingRecHitBuilder> TkTransientTrackingRecHitBuilderESProducer::produce(
    const TransientRecHitRecord& iRecord) {
  const StripClusterParameterEstimator* sp = nullptr;
  if (spToken_) {
    sp = &iRecord.get(*spToken_);
  }

  const PixelClusterParameterEstimator* pp = nullptr;
  if (ppToken_) {
    pp = &iRecord.get(*ppToken_);
  }

  const SiStripRecHitMatcher* mp = nullptr;
  if (mpToken_) {
    mp = &iRecord.get(*mpToken_);
  }

  if (computeCoarseLocalPositionFromDisk_)
    edm::LogWarning("TkTransientTrackingRecHitBuilderESProducer")
        << " The tracking rec hit positions and errors are not a persistent in data formats.\n"
        << " They are not available from disk.\n"
        << " However, TkTransientTrackingRecHitBuilderESProducer::ComputeCoarseLocalPositionFromDisk=True \n"
        << " will make the coarse estimation of this position/error available without track refit.\n"
        << " Position/error obtained from rechit with already defined position/error are not recomputed.\n"
        << " Position/error obtained from track refit are precise.";

  auto const& dd = iRecord.get(geomToken_);

  //For Phase2 upgrade
  if (p2OTToken_) {
    return std::make_unique<TkTransientTrackingRecHitBuilder>(&dd, pp, &iRecord.get(*p2OTToken_));
  }
  return std::make_unique<TkTransientTrackingRecHitBuilder>(&dd, pp, sp, mp, computeCoarseLocalPositionFromDisk_);
}

void TkTransientTrackingRecHitBuilderESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("ComponentName");
  desc.add<bool>("ComputeCoarseLocalPositionFromDisk");
  desc.add<std::string>("StripCPE")->setComment("Using \"Fake\" disables use of StripCPE");
  desc.add<std::string>("PixelCPE")->setComment("Using \"Fake\" disables use of PixelCPE");
  desc.add<std::string>("Matcher")->setComment("Using \"Fake\" disables use of SiStripRecHitMatcher");
  desc.add<std::string>("Phase2StripCPE", "")->setComment("Using empty string disables use of Phase2StripCPE");

  descriptions.addDefault(desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(TkTransientTrackingRecHitBuilderESProducer);
