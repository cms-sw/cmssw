#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "DataFormats/Portable/interface/PortableCollection.h"

#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"

#include <vector>
#include <algorithm>
#include <cmath>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class StackedModuleGeometryESProducer : public ESProducer {
  public:
    StackedModuleGeometryESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto c = setWhatProduced(this);
      geomToken_ = c.consumes();
      topoToken_ = c.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<reco::StackedModuleGeometryHost> produce(StackedModuleGeometryRecord const& iRecord) {
      auto const& geom = iRecord.get(geomToken_);
      auto const& topo = iRecord.get(topoToken_);

      struct ModuleInfo {
        const GeomDetUnit* detUnit;
        const GeomDetUnit* partnerDetUnit;
        DetId detId;
        DetId partnerDetId;
        DetId stackedDetId;
        uint8_t layer;
        bool isBarrel;
        bool isFwdEndcap;  // z > 0
        bool isPS;
        uint8_t category;  // 0 = barrel, 1 = endcap at z > 0, 2 = endcap at z < 0
      };

      std::vector<ModuleInfo> modules;
      modules.reserve(15000);  // Approximate number of OT stacked modules

      for (auto const& detUnit : geom.detUnits()) {
        DetId detId = detUnit->geographicalId();

        if (topo.stack(detId) == 0 || !topo.isLower(detId))
          continue;

        DetId partnerDetId = topo.partnerDetId(detId);
        DetId stackedDetId = topo.stack(detId);
        auto const* partnerDetUnit = geom.idToDetUnit(partnerDetId);

        if (!partnerDetUnit)
          continue;

        auto const& lowerPos = detUnit->position();
        bool isBarrel = (detId.subdetId() == StripSubdetector::TOB);
        bool isFwdEndcap = (lowerPos.z() > 0);
        bool isPS = (geom.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP) ||
                    (geom.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSS);
        uint8_t layer = topo.layer(detId);

        uint8_t category;
        if (isBarrel) {
          category = 0;
        } else if (isFwdEndcap) {
          category = 1;  // endcap at z > 0
        } else {
          category = 2;  // endcap at z < 0
        }

        modules.push_back(
            {detUnit, partnerDetUnit, detId, partnerDetId, stackedDetId, layer, isBarrel, isFwdEndcap, isPS, category});
      }

      // CA module order: OT module i gets CA index nPixelModules + i, and a new CA layer starts at
      // every (category, layer, PS/SS) transition. stable_sort keeps the TrackerGeometry order within
      // each group.
      std::stable_sort(modules.begin(), modules.end(), [](const ModuleInfo& a, const ModuleInfo& b) {
        if (a.category != b.category)
          return a.category < b.category;  // barrel, z > 0 endcap, z < 0 endcap
        if (a.layer != b.layer)
          return a.layer < b.layer;  // by layer within a category
        return a.isPS > b.isPS;      // PS modules before SS within a layer
      });

      uint32_t nModules = modules.size();

      auto hostProduct = std::make_unique<reco::StackedModuleGeometryHost>(cms::alpakatools::host(), nModules);
      auto view = hostProduct->view();

      for (uint32_t iModule = 0; iModule < nModules; ++iModule) {
        const auto& mod = modules[iModule];
        auto const& detUnit = mod.detUnit;
        auto const& partnerDetUnit = mod.partnerDetUnit;
        DetId detId = mod.detId;
        DetId stackedDetId = mod.stackedDetId;

        auto const& lowerPos = detUnit->position();
        auto const& upperPos = partnerDetUnit->position();

        // mm (geometry is in cm)
        float separation = (upperPos - lowerPos).mag() * 10.0f;

        bool isBarrel = mod.isBarrel;
        bool isPS = mod.isPS;
        bool isPSP = (geom.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP);
        bool isPSS = (geom.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSS);
        bool isFwdEndcap = mod.isFwdEndcap;

        // isFlipped: the topological "lower" sensor is farther from the beam line than the "upper" one.
        // A 3D distance would be wrong for tilted barrel modules, where the sensor at smaller r can sit
        // at larger |z|. Must be computed before tiltAngle, which depends on it.
        bool isFlipped;
        if (isBarrel) {
          isFlipped = (lowerPos.perp() > upperPos.perp());
        } else {
          isFlipped = (std::abs(lowerPos.z()) > std::abs(upperPos.z()));
        }

        // Tilt angle (rad): from the +r axis to the physical inner->outer direction, counterclockwise
        // in the (r, z) plane. Using the physical rather than the topological lower->upper direction
        // gives the same meaning for flipped and non-flipped modules: ~0 in the flat barrel, ~+pi/2
        // (-pi/2) in the +z (-z) endcap. The parallax correction and dPhiDr rely on this convention.
        float dz_phys, dr_phys;
        if (isFlipped) {
          // physical inner is "upper"
          dz_phys = lowerPos.z() - upperPos.z();
          dr_phys = lowerPos.perp() - upperPos.perp();  // > 0 for barrel
        } else {
          // physical inner is "lower"
          dz_phys = upperPos.z() - lowerPos.z();
          dr_phys = upperPos.perp() - lowerPos.perp();  // > 0 for barrel
        }
        float tiltAngle = std::atan2(dz_phys, dr_phys);

        // Unit vector along the physical inner->outer direction, consistent with tiltAngle.
        auto moduleAxisVec = isFlipped ? (lowerPos - upperPos) : (upperPos - lowerPos);
        auto moduleAxis = moduleAxisVec.basicVector() / moduleAxisVec.mag();

        auto localXInGlobal = detUnit->surface().toGlobal(LocalVector(1.0, 0.0, 0.0));

        // The width (lx_lower - lx_upper) needs local-x to point the same direction on both sensors,
        // the same-sign-y cut needs the same of local-y.
        {
          auto localXUpper = partnerDetUnit->surface().toGlobal(LocalVector(1.0, 0.0, 0.0));
          auto localYLower = detUnit->surface().toGlobal(LocalVector(0.0, 1.0, 0.0));
          auto localYUpper = partnerDetUnit->surface().toGlobal(LocalVector(0.0, 1.0, 0.0));
          auto localZLower = detUnit->surface().toGlobal(LocalVector(0.0, 0.0, 1.0));
          auto localZUpper = partnerDetUnit->surface().toGlobal(LocalVector(0.0, 0.0, 1.0));

          float dotX = localXInGlobal.x() * localXUpper.x() + localXInGlobal.y() * localXUpper.y() +
                       localXInGlobal.z() * localXUpper.z();
          float dotY =
              localYLower.x() * localYUpper.x() + localYLower.y() * localYUpper.y() + localYLower.z() * localYUpper.z();
          float dotZ =
              localZLower.x() * localZUpper.x() + localZLower.y() * localZUpper.y() + localZLower.z() * localZUpper.z();

          if (dotX < 0.9f || dotY < 0.9f) {
            edm::LogWarning("StackedModuleGeometry")
                << "LOCAL FRAME MISMATCH module " << iModule << " detId=" << detId.rawId() << " isBarrel=" << isBarrel
                << " isFlipped=" << isFlipped << " layer=" << (int)mod.layer << " dotX=" << dotX << " dotY=" << dotY
                << " dotZ=" << dotZ << " lower=(" << lowerPos.x() << "," << lowerPos.y() << "," << lowerPos.z() << ")"
                << " upper=(" << upperPos.x() << "," << upperPos.y() << "," << upperPos.z() << ")"
                << " localX_lower=(" << localXInGlobal.x() << "," << localXInGlobal.y() << "," << localXInGlobal.z()
                << ")"
                << " localX_upper=(" << localXUpper.x() << "," << localXUpper.y() << "," << localXUpper.z() << ")"
                << " localY_lower=(" << localYLower.x() << "," << localYLower.y() << "," << localYLower.z() << ")"
                << " localY_upper=(" << localYUpper.x() << "," << localYUpper.y() << "," << localYUpper.z() << ")";
          }
        }

        bool isFlat;
        const float threshold = std::cos(0.1f);  // not constexpr: std::cos is not a constant expression in clang
        if (isBarrel) {
          isFlat = std::abs(std::cos(tiltAngle)) > threshold;
        } else {
          isFlat = true;
        }

        view[iModule].detId() = detId.rawId();
        view[iModule].stackedDetId() = stackedDetId.rawId();
        view[iModule].moduleType() = (isPSP ? 0 : (isPSS ? 1 : 2));
        view[iModule].sensorSeparation() = separation;
        view[iModule].tiltAngle() = tiltAngle;
        view[iModule].sinTilt() = std::sin(tiltAngle);
        view[iModule].cosTilt() = std::cos(tiltAngle);
        view[iModule].isPS() = isPS;
        view[iModule].isFlipped() = isFlipped;
        view[iModule].isBarrel() = isBarrel;
        view[iModule].isFlat() = isFlat;
        view[iModule].isFwdEndcap() = isFwdEndcap;
        view[iModule].layer() = mod.layer;
        view[iModule].globalLowUpNormX() = moduleAxis.x();
        view[iModule].globalLowUpNormY() = moduleAxis.y();
        view[iModule].globalLowUpNormZ() = moduleAxis.z();
        view[iModule].localXInGlobalX() = localXInGlobal.x();
        view[iModule].localXInGlobalY() = localXInGlobal.y();
        view[iModule].localXInGlobalZ() = localXInGlobal.z();

        // Sensor surface frames (position + rotation): consumers propagate xerrLocal/yerrLocal to the
        // global covariance with SOAFrame::toGlobal instead of storing it in OTRecHitsSoA.
        auto fillFrame = [](auto const* d) {
          auto const& pos = d->surface().position();
          auto const& rot = d->surface().rotation();
          return SOAFrame<float>(static_cast<float>(pos.x()),
                                 static_cast<float>(pos.y()),
                                 static_cast<float>(pos.z()),
                                 SOARotation<float>(static_cast<float>(rot.xx()),
                                                    static_cast<float>(rot.xy()),
                                                    static_cast<float>(rot.xz()),
                                                    static_cast<float>(rot.yx()),
                                                    static_cast<float>(rot.yy()),
                                                    static_cast<float>(rot.yz()),
                                                    static_cast<float>(rot.zx()),
                                                    static_cast<float>(rot.zy()),
                                                    static_cast<float>(rot.zz())));
        };
        view[iModule].lowerSensorFrame() = fillFrame(detUnit);
        view[iModule].upperSensorFrame() = fillFrame(partnerDetUnit);
      }

      uint32_t nBarrel = 0, nBackward = 0, nForward = 0;
      for (const auto& mod : modules) {
        if (mod.category == 0)
          nBarrel++;
        else if (mod.category == 2)
          nBackward++;
        else
          nForward++;
      }
      edm::LogInfo("StackedModuleGeometry")
          << "Produced " << nModules << " stacked modules in CA order:\n"
          << "  Barrel: " << nBarrel << " (indices 0-" << (nBarrel - 1) << ")\n"
          << "  Endcap z > 0: " << nForward << " (indices " << nBarrel << "-" << (nBarrel + nForward - 1) << ")\n"
          << "  Endcap z < 0: " << nBackward << " (indices " << (nBarrel + nForward) << "-" << (nModules - 1) << ")";

      return hostProduct;
    }

  private:
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(StackedModuleGeometryESProducer);
