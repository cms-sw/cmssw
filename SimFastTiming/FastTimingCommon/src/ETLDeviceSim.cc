#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "SimFastTiming/FastTimingCommon/interface/ETLDeviceSim.h"
#include "DataFormats/Math/interface/GeantUnits.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

ETLDeviceSim::ETLDeviceSim(const edm::ParameterSet& pset, edm::ConsumesCollector iC)
    : geomToken_(iC.esConsumes()),
      geom_(nullptr),
      MIPPerMeV_(1.0 / pset.getParameter<double>("meVPerMIP")),
      bxTime_(pset.getParameter<double>("bxTime")),
      tofDelay_(pset.getParameter<double>("tofDelay")) {}

void ETLDeviceSim::getEventSetup(const edm::EventSetup& evs) { geom_ = &evs.getData(geomToken_); }

void ETLDeviceSim::getHitsResponse(const std::vector<std::tuple<int, uint32_t, float> >& hitRefs,
                                   const edm::Handle<edm::PSimHitContainer>& hits,
                                   mtd_digitizer::MTDSimHitDataAccumulator* simHitAccumulator,
                                   CLHEP::HepRandomEngine* hre) {
  using namespace geant_units::operators;

  //loop over sorted hits
  const int nchits = hitRefs.size();
  for (int i = 0; i < nchits; ++i) {
    const int hitidx = std::get<0>(hitRefs[i]);
    const uint32_t id = std::get<1>(hitRefs[i]);
    const MTDDetId detId(id);

    // Safety check (this should never happen, it should be an exception
    if (detId.det() != DetId::Forward || detId.mtdSubDetector() != 2) {
      throw cms::Exception("ETLDeviceSim")
          << "got a DetId that was not ETL: Det = " << detId.det() << "  subDet = " << detId.mtdSubDetector();
    }

    if (id == 0)
      continue;  // to be ignored at RECO level

    ETLDetId etlid(detId);
    DetId geoId = ETLDetId(etlid.mtdSide(), etlid.mtdRR(), etlid.module(), etlid.modType());
    const MTDGeomDet* thedet = geom_->idToDet(geoId);
    if (thedet == nullptr) {
      throw cms::Exception("ETLDeviceSim") << "GeographicalID: " << std::hex << geoId.rawId() << " (" << detId.rawId()
                                           << ") is invalid!" << std::dec << std::endl;
    }
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

    const float toa = std::get<2>(hitRefs[i]) + tofDelay_;
    const PSimHit& hit = hits->at(hitidx);
    const float charge = convertGeVToMeV(hit.energyLoss()) * MIPPerMeV_;

    // calculate the simhit row and column
    const auto& position = hit.localPosition();
    // ETL is already in module-local coordinates so just scale to cm from mm
    Local3DPoint simscaled(convertMmToCm(position.x()), convertMmToCm(position.y()), convertMmToCm(position.z()));
    //The following lines check whether the pixel point is actually out of the active area.
    //If that is the case it simply ignores the point but in the future some more sophisticated function could be applied.
    if (!topo.isInPixel(simscaled)) {
      continue;
    }
    const auto& thepixel = topo.pixel(simscaled);
    const uint8_t row(thepixel.first), col(thepixel.second);

    auto simHitIt =
        simHitAccumulator->emplace(mtd_digitizer::MTDCellId(id, row, col), mtd_digitizer::MTDCellInfo()).first;

    // Accumulate in 15 buckets of 25ns (9 pre-samples, 1 in-time, 5 post-samples)
    const int itime = std::floor(toa / bxTime_) + 9;
    if (itime < 0 || itime > 14)
      continue;

    // Check if time index is ok and store energy
    if (itime >= (int)simHitIt->second.hit_info[0].size())
      continue;

    (simHitIt->second).hit_info[0][itime] += charge;

    // Store the time of the first SimHit in the right DataFrame bucket
    const float tof = toa - (itime - 9) * bxTime_;

    if ((simHitIt->second).hit_info[1][itime] == 0. || tof < (simHitIt->second).hit_info[1][itime]) {
      (simHitIt->second).hit_info[1][itime] = tof;
    }
  }
}
