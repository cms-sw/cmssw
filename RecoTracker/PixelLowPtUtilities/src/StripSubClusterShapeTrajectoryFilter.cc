#include "RecoTracker/PixelLowPtUtilities/interface/StripSubClusterShapeTrajectoryFilter.h"

#include <map>
#include <set>
#include <algorithm>

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

#ifdef StripSubClusterShapeFilterBase_COUNTERS
#define INC_COUNTER(X) X++;
#else
#define INC_COUNTER(X)
#endif
namespace {
  class SlidingPeakFinder {
  public:
    SlidingPeakFinder(unsigned int size) : size_(size), half_((size + 1) / 2) {}

    template <typename Test>
    bool apply(const uint8_t *x,
               const uint8_t *begin,
               const uint8_t *end,
               const Test &test,
               bool verbose = false,
               int firststrip = 0) {
      const uint8_t *ileft = (x != begin) ? std::min_element(x - 1, x + half_) : begin - 1;
      const uint8_t *iright = ((x + size_) < end) ? std::min_element(x + half_, std::min(x + size_ + 1, end)) : end;
      uint8_t left = (ileft < begin ? 0 : *ileft);
      uint8_t right = (iright >= end ? 0 : *iright);
      uint8_t center = *std::max_element(x, std::min(x + size_, end));
      uint8_t maxmin = std::max(left, right);
      if (maxmin < center) {
        bool ret = test(center, maxmin);
        if (ret) {
          ret = test(ileft, iright, begin, end);
        }
        return ret;
      } else {
        return false;
      }
    }

    template <typename V, typename Test>
    bool apply(const V &ampls, const Test &test, bool verbose = false, int firststrip = 0) {
      const uint8_t *begin = &*ampls.begin();
      const uint8_t *end = &*ampls.end();
      for (const uint8_t *x = begin; x < end - (half_ - 1); ++x) {
        if (apply(x, begin, end, test, verbose, firststrip)) {
          return true;
        }
      }
      return false;
    }

  private:
    unsigned int size_, half_;
  };

  struct PeakFinderTest {
    PeakFinderTest(float mip,
                   uint32_t detid,
                   uint32_t firstStrip,
                   const SiStripNoises *theNoise,
                   float seedCutMIPs,
                   float seedCutSN,
                   float subclusterCutMIPs,
                   float subclusterCutSN)
        : mip_(mip),
          detid_(detid),
          firstStrip_(firstStrip),
          noiseObj_(theNoise),
          noises_(theNoise->getRange(detid)),
          subclusterCutMIPs_(subclusterCutMIPs),
          sumCut_(subclusterCutMIPs_ * mip_),
          subclusterCutSN2_(subclusterCutSN * subclusterCutSN) {
      cut_ = std::min<float>(seedCutMIPs * mip, seedCutSN * noiseObj_->getNoise(firstStrip + 1, noises_));
    }

    bool operator()(uint8_t max, uint8_t min) const { return max - min > cut_; }
    bool operator()(const uint8_t *left, const uint8_t *right, const uint8_t *begin, const uint8_t *end) const {
      int yleft = (left < begin ? 0 : *left);
      int yright = (right >= end ? 0 : *right);
      float sum = 0.0;
      int maxval = 0;
      float noise = 0;
      for (const uint8_t *x = left + 1; x < right; ++x) {
        int baseline = (yleft * int(right - x) + yright * int(x - left)) / int(right - left);
        sum += int(*x) - baseline;
        noise += std::pow(noiseObj_->getNoise(firstStrip_ + int(x - begin), noises_), 2);
        maxval = std::max(maxval, int(*x) - baseline);
      }
      if (sum > sumCut_ && sum * sum > noise * subclusterCutSN2_)
        return true;
      return false;
    }

  private:
    float mip_;
    unsigned int detid_;
    int firstStrip_;
    const SiStripNoises *noiseObj_;
    SiStripNoises::Range noises_;
    uint8_t cut_;
    float subclusterCutMIPs_, sumCut_, subclusterCutSN2_;
  };

}  // namespace

StripSubClusterShapeFilterBase::StripSubClusterShapeFilterBase(const edm::ParameterSet &iCfg,
                                                               edm::ConsumesCollector &iC)
    : topoToken_(iC.esConsumes<TrackerTopology, TrackerTopologyRcd>()),
      csfToken_(
          iC.esConsumes<ClusterShapeHitFilter, CkfComponentsRecord>(edm::ESInputTag("", "ClusterShapeHitFilter"))),
      geomToken_(iC.esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      stripNoiseToken_(iC.esConsumes<SiStripNoises, SiStripNoisesRcd>()),
      label_(iCfg.getUntrackedParameter<std::string>("label", "")),
      maxNSat_(iCfg.getParameter<uint32_t>("maxNSat")),
      trimMaxADC_(iCfg.getParameter<double>("trimMaxADC")),
      trimMaxFracTotal_(iCfg.getParameter<double>("trimMaxFracTotal")),
      trimMaxFracNeigh_(iCfg.getParameter<double>("trimMaxFracNeigh")),
      maxTrimmedSizeDiffPos_(iCfg.getParameter<double>("maxTrimmedSizeDiffPos")),
      maxTrimmedSizeDiffNeg_(iCfg.getParameter<double>("maxTrimmedSizeDiffNeg")),
      subclusterWindow_(iCfg.getParameter<double>("subclusterWindow")),
      seedCutMIPs_(iCfg.getParameter<double>("seedCutMIPs")),
      seedCutSN_(iCfg.getParameter<double>("seedCutSN")),
      subclusterCutMIPs_(iCfg.getParameter<double>("subclusterCutMIPs")),
      subclusterCutSN_(iCfg.getParameter<double>("subclusterCutSN"))
#ifdef StripSubClusterShapeFilterBase_COUNTERS
      ,
      called_(0),
      saturated_(0),
      test_(0),
      passTrim_(0),
      failTooLarge_(0),
      passSC_(0),
      failTooNarrow_(0)
#endif
{
  const edm::ParameterSet &iLM = iCfg.getParameter<edm::ParameterSet>("layerMask");
  if (not iLM.empty()) {
    const char *ndets[4] = {"TIB", "TID", "TOB", "TEC"};
    const int idets[4] = {3, 4, 5, 6};
    for (unsigned int i = 0; i < 4; ++i) {
      if (iLM.existsAs<bool>(ndets[i])) {
        std::fill(layerMask_[idets[i]].begin(), layerMask_[idets[i]].end(), iLM.getParameter<bool>(ndets[i]));
      } else {
        layerMask_[idets[i]][0] = 2;
        std::fill(layerMask_[idets[i]].begin() + 1, layerMask_[idets[i]].end(), 0);
        for (uint32_t lay : iLM.getParameter<std::vector<uint32_t>>(ndets[i])) {
          layerMask_[idets[i]][lay] = 1;
        }
      }
    }
  } else {
    for (auto &arr : layerMask_) {
      std::fill(arr.begin(), arr.end(), 1);
    }
  }
}

StripSubClusterShapeFilterBase::~StripSubClusterShapeFilterBase() {
#if 0
  std::cout << "StripSubClusterShapeFilterBase " << label_ << ": called        " << called_ << std::endl;
  std::cout << "StripSubClusterShapeFilterBase " << label_ << ": saturated     " << saturated_ << std::endl;
  std::cout << "StripSubClusterShapeFilterBase " << label_ << ": test          " << test_ << std::endl;
  std::cout << "StripSubClusterShapeFilterBase " << label_ << ": failTooNarrow " << failTooNarrow_ << std::endl;
  std::cout << "StripSubClusterShapeFilterBase " << label_ << ": passTrim      " << passTrim_ << std::endl;
  std::cout << "StripSubClusterShapeFilterBase " << label_ << ": passSC        " << passSC_ << std::endl;
  std::cout << "StripSubClusterShapeFilterBase " << label_ << ": failTooLarge  " << failTooLarge_ << std::endl;
#endif
}

void StripSubClusterShapeFilterBase::fillPSetDescription(edm::ParameterSetDescription &iDesc) {
  iDesc.addUntracked<std::string>("label", "");
  iDesc.add<uint32_t>("maxNSat", 3);
  iDesc.add<double>("trimMaxADC", 30.);
  iDesc.add<double>("trimMaxFracTotal", .15);
  iDesc.add<double>("trimMaxFracNeigh", .25);
  iDesc.add<double>("maxTrimmedSizeDiffPos", .7);
  iDesc.add<double>("maxTrimmedSizeDiffNeg", 1.);
  iDesc.add<double>("subclusterWindow", .7);
  iDesc.add<double>("seedCutMIPs", .35);
  iDesc.add<double>("seedCutSN", 7.);
  iDesc.add<double>("subclusterCutMIPs", .45);
  iDesc.add<double>("subclusterCutSN", 12.);

  edm::ParameterSetDescription psdLM;
  psdLM.setAllowAnything();
  iDesc.add<edm::ParameterSetDescription>("layerMask", psdLM);
}

bool StripSubClusterShapeFilterBase::testLastHit(const TrackingRecHit *hit,
                                                 const TrajectoryStateOnSurface &tsos,
                                                 bool mustProject) const {
  return testLastHit(hit, tsos.globalPosition(), tsos.globalDirection(), mustProject);
}
bool StripSubClusterShapeFilterBase::testLastHit(const TrackingRecHit *hit,
                                                 const GlobalPoint &gpos,
                                                 const GlobalVector &gdir,
                                                 bool mustProject) const {
  const TrackerSingleRecHit *stripHit = nullptr;
  if (typeid(*hit) == typeid(SiStripMatchedRecHit2D)) {
    const SiStripMatchedRecHit2D &mhit = static_cast<const SiStripMatchedRecHit2D &>(*hit);
    SiStripRecHit2D mono = mhit.monoHit();
    SiStripRecHit2D stereo = mhit.stereoHit();
    return testLastHit(&mono, gpos, gdir, true) && testLastHit(&stereo, gpos, gdir, true);
  } else if (typeid(*hit) == typeid(ProjectedSiStripRecHit2D)) {
    const ProjectedSiStripRecHit2D &mhit = static_cast<const ProjectedSiStripRecHit2D &>(*hit);
    const SiStripRecHit2D &orig = mhit.originalHit();
    return testLastHit(&orig, gpos, gdir, true);
  } else if ((stripHit = dynamic_cast<const TrackerSingleRecHit *>(hit)) != nullptr) {
    DetId detId = hit->geographicalId();

    if (layerMask_[detId.subdetId()][0] == 0) {
      return true;  // no filtering here
    } else if (layerMask_[detId.subdetId()][0] == 2) {
      unsigned int ilayer = theTopology->layer(detId);
      if (layerMask_[detId.subdetId()][ilayer] == 0) {
        return true;  // no filtering here
      }
    }

    const GeomDet *det = theTracker->idToDet(detId);
    LocalVector ldir = det->toLocal(gdir);
    LocalPoint lpos = det->toLocal(gpos);
    if (mustProject) {
      lpos -= ldir * lpos.z() / ldir.z();
    }
    int hitStrips;
    float hitPredPos;
    const SiStripCluster &cluster = stripHit->stripCluster();
    bool usable = theFilter->getSizes(detId, cluster, lpos, ldir, hitStrips, hitPredPos);
    if (!usable)
      return true;

    INC_COUNTER(called_)
    const auto &ampls = cluster.amplitudes();

    // pass-through of trivial case
    if (std::abs(hitPredPos) < 1.5f && hitStrips <= 2) {
      return true;
    }

    if (!cluster.isFromApprox()) {
      // compute number of consecutive saturated strips
      // (i.e. with adc count >= 254, see SiStripCluster class for documentation)
      unsigned int thisSat = (ampls[0] >= 254), maxSat = thisSat;
      for (unsigned int i = 1, n = ampls.size(); i < n; ++i) {
        if (ampls[i] >= 254) {
          thisSat++;
        } else if (thisSat > 0) {
          maxSat = std::max<int>(maxSat, thisSat);
          thisSat = 0;
        }
      }
      if (thisSat > 0) {
        maxSat = std::max<int>(maxSat, thisSat);
      }
      if (maxSat >= maxNSat_) {
        INC_COUNTER(saturated_)
        return true;
      }

      // trimming
      INC_COUNTER(test_)
      unsigned int hitStripsTrim = ampls.size();
      int sum = std::accumulate(ampls.begin(), ampls.end(), 0);
      uint8_t trimCut = std::min<uint8_t>(trimMaxADC_, std::floor(trimMaxFracTotal_ * sum));
      auto begin = ampls.begin();
      auto last = ampls.end() - 1;
      while (hitStripsTrim > 1 && (*begin < std::max<uint8_t>(trimCut, trimMaxFracNeigh_ * (*(begin + 1))))) {
        hitStripsTrim--;
        ++begin;
      }
      while (hitStripsTrim > 1 && (*last < std::max<uint8_t>(trimCut, trimMaxFracNeigh_ * (*(last - 1))))) {
        hitStripsTrim--;
        --last;
      }

      if (hitStripsTrim < std::floor(std::abs(hitPredPos) - maxTrimmedSizeDiffNeg_)) {
        INC_COUNTER(failTooNarrow_)
        return false;
      } else if (hitStripsTrim <= std::ceil(std::abs(hitPredPos) + maxTrimmedSizeDiffPos_)) {
        INC_COUNTER(passTrim_)
        return true;
      }

      const StripGeomDetUnit *stripDetUnit = dynamic_cast<const StripGeomDetUnit *>(det);
      if (det == nullptr) {
        edm::LogError("Strip not a StripGeomDetUnit?") << " on " << detId.rawId() << "\n";
        return true;
      }

      float mip = 3.9 / (sistrip::MeVperADCStrip /
                         stripDetUnit->surface().bounds().thickness());  // 3.9 MeV/cm = ionization in silicon
      float mipnorm = mip / std::abs(ldir.z());
      ::SlidingPeakFinder pf(std::max<int>(2, std::ceil(std::abs(hitPredPos) + subclusterWindow_)));
      ::PeakFinderTest test(mipnorm,
                            detId(),
                            cluster.firstStrip(),
                            &*theNoise,
                            seedCutMIPs_,
                            seedCutSN_,
                            subclusterCutMIPs_,
                            subclusterCutSN_);
      if (pf.apply(cluster.amplitudes(), test)) {
        INC_COUNTER(passSC_)
        return true;
      } else {
        INC_COUNTER(failTooLarge_)
        return false;
      }
    } else {
      return cluster.filter();
    }
  }
  return true;
}

void StripSubClusterShapeFilterBase::setEventBase(const edm::Event &event, const edm::EventSetup &es) {
  // Get tracker geometry
  theTracker = es.getHandle(geomToken_);
  theFilter = es.getHandle(csfToken_);

  //Retrieve tracker topology from geometry
  theTopology = es.getHandle(topoToken_);
  theNoise = es.getHandle(stripNoiseToken_);
}

/*****************************************************************************/
bool StripSubClusterShapeTrajectoryFilter::toBeContinued(Trajectory &trajectory) const {
  throw cms::Exception("toBeContinued(Traj) instead of toBeContinued(TempTraj)");
}

/*****************************************************************************/
bool StripSubClusterShapeTrajectoryFilter::testLastHit(const TrajectoryMeasurement &last) const {
  const TrackingRecHit *hit = last.recHit()->hit();
  if (!last.updatedState().isValid())
    return true;
  if (hit == nullptr || !hit->isValid())
    return true;
  if (hit->geographicalId().subdetId() < SiStripDetId::TIB)
    return true;  // we look only at strips for now
  return testLastHit(hit, last.updatedState(), false);
}

/*****************************************************************************/
bool StripSubClusterShapeTrajectoryFilter::toBeContinued(TempTrajectory &trajectory) const {
  const TempTrajectory::DataContainer &tms = trajectory.measurements();
  return testLastHit(*tms.rbegin());
}

/*****************************************************************************/
bool StripSubClusterShapeTrajectoryFilter::qualityFilter(const Trajectory &trajectory) const {
  const Trajectory::DataContainer &tms = trajectory.measurements();
  return testLastHit(*tms.rbegin());
}

/*****************************************************************************/
bool StripSubClusterShapeTrajectoryFilter::qualityFilter(const TempTrajectory &trajectory) const {
  const TempTrajectory::DataContainer &tms = trajectory.measurements();
  return testLastHit(*tms.rbegin());
}

/*****************************************************************************/
StripSubClusterShapeSeedFilter::StripSubClusterShapeSeedFilter(const edm::ParameterSet &iConfig,
                                                               edm::ConsumesCollector &iC)
    : StripSubClusterShapeFilterBase(iConfig, iC),
      filterAtHelixStage_(iConfig.getParameter<bool>("FilterAtHelixStage"))

{
  if (filterAtHelixStage_)
    edm::LogError("Configuration")
        << "StripSubClusterShapeSeedFilter: FilterAtHelixStage is not yet working correctly.\n";
}

/*****************************************************************************/
bool StripSubClusterShapeSeedFilter::compatible(const TrajectoryStateOnSurface &tsos,
                                                SeedingHitSet::ConstRecHitPointer thit) const {
  if (filterAtHelixStage_)
    return true;
  const TrackingRecHit *hit = thit->hit();
  if (hit == nullptr || !hit->isValid())
    return true;
  if (hit->geographicalId().subdetId() < SiStripDetId::TIB)
    return true;  // we look only at strips for now
  return testLastHit(hit, tsos, false);
}

bool StripSubClusterShapeSeedFilter::compatible(const SeedingHitSet &hits,
                                                const GlobalTrajectoryParameters &helixStateAtVertex,
                                                const FastHelix &helix) const {
  if (!filterAtHelixStage_)
    return true;

  if (!helix.isValid()              //check still if it's a straight line, which are OK
      && !helix.circle().isLine())  //complain if it's not even a straight line
    edm::LogWarning("InvalidHelix") << "StripSubClusterShapeSeedFilter helix is not valid, result is bad";

  float xc = helix.circle().x0(), yc = helix.circle().y0();

  GlobalPoint vertex = helixStateAtVertex.position();
  GlobalVector momvtx = helixStateAtVertex.momentum();
  float x0 = vertex.x(), y0 = vertex.y();
  for (unsigned int i = 0, n = hits.size(); i < n; ++i) {
    auto const &hit = *hits[i];
    if (hit.geographicalId().subdetId() < SiStripDetId::TIB)
      continue;

    GlobalPoint pos = hit.globalPosition();
    float x1 = pos.x(), y1 = pos.y(), dx1 = x1 - xc, dy1 = y1 - yc;

    // now figure out the proper tangent vector
    float perpx = -dy1, perpy = dx1;
    if (perpx * (x1 - x0) + perpy * (y1 - y0) < 0) {
      perpy = -perpy;
      perpx = -perpx;
    }

    // now normalize (perpx, perpy, 1.0) to unity
    float pnorm = 1.0 / std::sqrt(perpx * perpx + perpy * perpy + 1);
    GlobalVector gdir(perpx * pnorm, perpy * pnorm, (momvtx.z() > 0 ? pnorm : -pnorm));

    if (!testLastHit(&hit, pos, gdir)) {
      return false;  // not yet
    }
  }
  return true;
}
