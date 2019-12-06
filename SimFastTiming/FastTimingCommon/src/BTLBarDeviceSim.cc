#include "SimFastTiming/FastTimingCommon/interface/BTLBarDeviceSim.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"

#include "CLHEP/Random/RandGaussQ.h"

BTLBarDeviceSim::BTLBarDeviceSim(const edm::ParameterSet& pset)
    : geom_(nullptr),
      topo_(nullptr),
      bxTime_(pset.getParameter<double>("bxTime")),
      LightYield_(pset.getParameter<double>("LightYield")),
      LightCollEff_(pset.getParameter<double>("LightCollectionEff")),
      LightCollSlopeR_(pset.getParameter<double>("LightCollectionSlopeR")),
      LightCollSlopeL_(pset.getParameter<double>("LightCollectionSlopeL")),
      PDE_(pset.getParameter<double>("PhotonDetectionEff")) {}

void BTLBarDeviceSim::getEventSetup(const edm::EventSetup& evs) {
  edm::ESHandle<MTDGeometry> geom;
  evs.get<MTDDigiGeometryRecord>().get(geom);
  geom_ = geom.product();

  edm::ESHandle<MTDTopology> mtdTopo;
  evs.get<MTDTopologyRcd>().get(mtdTopo);
  topo_ = mtdTopo.product();
}

void BTLBarDeviceSim::getHitsResponse(const std::vector<std::tuple<int, uint32_t, float> >& hitRefs,
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
    const int boundRef = btlid.modulesPerType(static_cast<BTLDetId::CrysLayout>(topo_->getMTDTopologyMode()));
    DetId geoId = BTLDetId(btlid.mtdSide(), btlid.mtdRR(), btlid.module() + boundRef * (btlid.modType() - 1), 0, 1);
    const MTDGeomDet* thedet = geom_->idToDet(geoId);

    if (thedet == nullptr) {
      throw cms::Exception("BTLBarDeviceSim") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
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
      edm::LogWarning("BTLBarDeviceSim") << "BTLDetId (row,column): (" << btlid.row(topo.nrows()) << ','
                                         << btlid.column(topo.nrows()) << ") is not equal to "
                                         << "topology (row,column): (" << uint32_t(row) << ',' << uint32_t(col)
                                         << "), overriding to detid";
      row = btlid.row(topo.nrows());
      col = btlid.column(topo.nrows());
    }

    // --- Store the detector element ID as a key of the MTDSimHitDataAccumulator map
    auto simHitIt =
        simHitAccumulator->emplace(mtd_digitizer::MTDCellId(id, row, col), mtd_digitizer::MTDCellInfo()).first;

    // --- Get the simHit energy and convert it from MeV to photo-electrons
    float Npe = 1000. * hit.energyLoss() * LightYield_ * LightCollEff_ * PDE_;

    // --- Calculate the light propagation time to the crystal bases (labeled L and R)
    double distR = 0.5 * topo.pitch().second - 0.1 * hit.localPosition().y();
    double distL = 0.5 * topo.pitch().second + 0.1 * hit.localPosition().y();

    // This is for the layouts with bars along phi
    if (topo_->getMTDTopologyMode() == (int)BTLDetId::CrysLayout::bar ||
        topo_->getMTDTopologyMode() == (int)BTLDetId::CrysLayout::barphiflat) {
      distR = 0.5 * topo.pitch().first - 0.1 * hit.localPosition().x();
      distL = 0.5 * topo.pitch().first + 0.1 * hit.localPosition().x();
    }

    double tR = std::get<2>(hitRef) + LightCollSlopeR_ * distR;
    double tL = std::get<2>(hitRef) + LightCollSlopeL_ * distL;

    // --- Accumulate in 15 buckets of 25ns (9 pre-samples, 1 in-time, 5 post-samples)
    const int iBXR = std::floor(tR / bxTime_) + mtd_digitizer::kInTimeBX;
    const int iBXL = std::floor(tL / bxTime_) + mtd_digitizer::kInTimeBX;

    // --- Right side
    if (iBXR > 0 && iBXR < mtd_digitizer::kNumberOfBX) {
      // Accumulate the energy of simHits in the same crystal
      (simHitIt->second).hit_info[0][iBXR] += Npe;

      // Store the time of the first SimHit in the i-th BX
      if ((simHitIt->second).hit_info[1][iBXR] == 0 || tR < (simHitIt->second).hit_info[1][iBXR])
        (simHitIt->second).hit_info[1][iBXR] = tR - (iBXR - mtd_digitizer::kInTimeBX) * bxTime_;
    }

    // --- Left side
    if (iBXL > 0 && iBXL < mtd_digitizer::kNumberOfBX) {
      // Accumulate the energy of simHits in the same crystal
      (simHitIt->second).hit_info[2][iBXL] += Npe;

      // Store the time of the first SimHit in the i-th BX
      if ((simHitIt->second).hit_info[3][iBXL] == 0 || tL < (simHitIt->second).hit_info[3][iBXL])
        (simHitIt->second).hit_info[3][iBXL] = tL - (iBXL - mtd_digitizer::kInTimeBX) * bxTime_;
    }

  }  // hitRef loop
}
