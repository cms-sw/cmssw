#include "SimFastTiming/FastTimingCommon/interface/BTLTileDeviceSim.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

#include "CLHEP/Random/RandGaussQ.h"

BTLTileDeviceSim::BTLTileDeviceSim(const edm::ParameterSet& pset)
    : geom_(nullptr),
      bxTime_(pset.getParameter<double>("bxTime")),
      LightYield_(pset.getParameter<double>("LightYield")),
      LightCollEff_(pset.getParameter<double>("LightCollectionEff")),
      LightCollTime_(pset.getParameter<double>("LightCollectionTime")),
      smearLightCollTime_(pset.getParameter<double>("smearLightCollectionTime")),
      PDE_(pset.getParameter<double>("PhotonDetectionEff")) {}

void BTLTileDeviceSim::getEventSetup(const edm::EventSetup& evs) {
  edm::ESHandle<MTDGeometry> geom;
  evs.get<MTDDigiGeometryRecord>().get(geom);
  geom_ = geom.product();
}

void BTLTileDeviceSim::getHitsResponse(const std::vector<std::tuple<int, uint32_t, float> >& hitRefs,
                                       const edm::Handle<edm::PSimHitContainer>& hits,
                                       mtd_digitizer::MTDSimHitDataAccumulator* simHitAccumulator,
                                       CLHEP::HepRandomEngine* hre) {
  //loop over sorted simHits
  for (auto const& hitRef : hitRefs) {
    const int hitidx = std::get<0>(hitRef);
    const uint32_t id = std::get<1>(hitRef);
    const MTDDetId detId(id);
    const PSimHit& hit = hits->at(hitidx);

    // --- Safety check on the detector ID
    if (detId.det() != DetId::Forward || detId.mtdSubDetector() != 1)
      continue;

    if (id == 0)
      continue;  // to be ignored at RECO level

    BTLDetId btlid(detId);
    const int boundRef = BTLDetId::kTypeBoundariesReference[1];
    DetId geoId = BTLDetId(btlid.mtdSide(), btlid.mtdRR(), btlid.module() + boundRef * (btlid.modType() - 1), 0, 1);
    const MTDGeomDet* thedet = geom_->idToDet(geoId);

    if (thedet == nullptr) {
      throw cms::Exception("BTLTileDeviceSim") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                               << detId.rawId() << ") is invalid!" << std::dec << std::endl;
    }
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());
    // calculate the simhit row and column
    const auto& pentry = hit.entryPoint();
    Local3DPoint simscaled(0.1 * pentry.x(), 0.1 * pentry.y(), 0.1 * pentry.z());  // mm -> cm here is the switch
    // translate from crystal-local coordinates to module-local coordinates to get the row and column
    simscaled = topo.pixelToModuleLocalPoint(simscaled, btlid.row(topo.nrows()), btlid.column(topo.nrows()));
    const auto& thepixel = topo.pixel(simscaled);
    uint8_t row(thepixel.first), col(thepixel.second);

    if (btlid.row(topo.nrows()) != row || btlid.column(topo.nrows()) != col) {
      edm::LogWarning("BTLTileDeviceSim")
          << "BTLDetId (row,column): (" << btlid.row(topo.nrows()) << ',' << btlid.column(topo.nrows())
          << ") is not equal to "
          << "topology (row,column): (" << uint32_t(row) << ',' << uint32_t(col) << "), overriding to detid";
      row = btlid.row(topo.nrows());
      col = btlid.column(topo.nrows());
    }

    // --- Store the detector element ID as a key of the MTDSimHitDataAccumulator map
    auto simHitIt =
        simHitAccumulator->emplace(mtd_digitizer::MTDCellId(id, row, col), mtd_digitizer::MTDCellInfo()).first;

    // --- Get the simHit energy and convert it from MeV to photo-electrons
    float Npe = 1000. * hit.energyLoss() * LightYield_ * LightCollEff_ * PDE_;

    // --- Get the simHit time of arrival and add the light collection time
    float toa = std::get<2>(hitRef) + LightCollTime_;

    if (smearLightCollTime_ > 0.)
      toa += CLHEP::RandGaussQ::shoot(hre, 0., smearLightCollTime_);

    // Accumulate in 15 buckets of 25ns (9 pre-samples, 1 in-time, 5 post-samples)
    const int iBX = std::floor(toa / bxTime_) + mtd_digitizer::kInTimeBX;
    if (iBX < 0 || iBX >= mtd_digitizer::kNumberOfBX)
      continue;

    (simHitIt->second).hit_info[0][iBX] += Npe;

    // --- Store the time of the first SimHit in the i-th BX
    if ((simHitIt->second).hit_info[1][iBX] == 0 || toa < (simHitIt->second).hit_info[1][iBX])
      (simHitIt->second).hit_info[1][iBX] = toa - (iBX - mtd_digitizer::kInTimeBX) * bxTime_;

  }  // hitRef loop
}
