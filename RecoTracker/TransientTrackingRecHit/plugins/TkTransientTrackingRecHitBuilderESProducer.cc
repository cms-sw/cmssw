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
  const std::string pname_;
  std::optional<edm::ESGetToken<StripClusterParameterEstimator, TkStripCPERecord>> spToken_;
  std::optional<edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord>> ppToken_;
  std::optional<edm::ESGetToken<SiStripRecHitMatcher, TkStripCPERecord>> mpToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  std::optional<edm::ESGetToken<ClusterParameterEstimator<Phase2TrackerCluster1D>, TkPhase2OTCPERecord>> p2OTToken_;
  bool const computeCoarseLocalPositionFromDisk_;
};

using namespace edm;

TkTransientTrackingRecHitBuilderESProducer::TkTransientTrackingRecHitBuilderESProducer(const edm::ParameterSet& p)
    : pname_(p.getParameter<std::string>("PixelCPE")),
      computeCoarseLocalPositionFromDisk_(p.getParameter<bool>("ComputeCoarseLocalPositionFromDisk")) {
  std::string const myname = p.getParameter<std::string>("ComponentName");
  auto c = setWhatProduced(this, myname);
  geomToken_ = c.consumes();

  std::string const sname = p.getParameter<std::string>("StripCPE");
  if (sname != "Fake") {
    spToken_ = c.consumes(edm::ESInputTag("", sname));
  }

  if (pname_ != "Fake") {
    ppToken_ = c.consumes(edm::ESInputTag("", pname_));
  }

  auto const mname = p.getParameter<std::string>("Matcher");
  if (mname != "Fake") {
    mpToken_ = c.consumes(edm::ESInputTag("", mname));
  }

  auto const P2otname = p.getParameter<std::string>("Phase2StripCPE");
  if (!P2otname.empty()) {
    p2OTToken_ = c.consumes(edm::ESInputTag("", P2otname));
  }
}

std::unique_ptr<TransientTrackingRecHitBuilder> TkTransientTrackingRecHitBuilderESProducer::produce(
    const TransientRecHitRecord& iRecord) {
  if (pname_ == "PixelCPEFast") {
    edm::LogWarning("TkTransientTrackingRecHitBuilderESProducer")
        << "\n\t\t WARNING!\n 'PixelCPEFast' has been chosen as PixelCPE choice.\n"
        << " Track angles will NOT be used in the CPE estimation!\n";
  }

  const StripClusterParameterEstimator* sp = nullptr;
  if (spToken_ && !p2OTToken_) {  // no strips in Phase-2
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
  desc.add<std::string>("ComponentName", "Fake");
  desc.add<bool>("ComputeCoarseLocalPositionFromDisk", false);
  desc.add<std::string>("StripCPE", "Fake")->setComment("Using \"Fake\" disables use of StripCPE");
  desc.add<std::string>("PixelCPE", "Fake")->setComment("Using \"Fake\" disables use of PixelCPE");
  desc.add<std::string>("Matcher", "Fake")->setComment("Using \"Fake\" disables use of SiStripRecHitMatcher");
  desc.add<std::string>("Phase2StripCPE", "")->setComment("Using empty string disables use of Phase2StripCPE");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(TkTransientTrackingRecHitBuilderESProducer);
