
// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      BtlSimHitsValidation
//
/**\class BtlSimHitsValidation BtlSimHitsValidation.cc Validation/MtdValidation/plugins/BtlSimHitsValidation.cc

 Description: BTL SIM hits validation

 Implementation:
     [Notes on implementation]
*/

#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/ValidHandle.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

struct MTDHit {
  float energy;
  float time;
  float x;
  float y;
  float z;
};

class BtlSimHitsValidation : public DQMEDAnalyzer {
public:
  explicit BtlSimHitsValidation(const edm::ParameterSet&);
  ~BtlSimHitsValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ------------ member data ------------

  const std::string folder_;

  const float hitMinEnergy_;

  edm::EDGetTokenT<edm::PSimHitContainer> btlSimHitsToken_;

  // --- histograms declaration

  MonitorElement* meNhits_;
  MonitorElement* meNtrkPerCell_;

  MonitorElement* meHitEnergy_;
  MonitorElement* meHitTime_;

  MonitorElement* meHitXlocal_;
  MonitorElement* meHitYlocal_;
  MonitorElement* meHitZlocal_;

  MonitorElement* meOccupancy_;

  MonitorElement* meHitX_;
  MonitorElement* meHitY_;
  MonitorElement* meHitZ_;
  MonitorElement* meHitPhi_;
  MonitorElement* meHitEta_;

  MonitorElement* meHitTvsE_;
  MonitorElement* meHitEvsPhi_;
  MonitorElement* meHitEvsEta_;
  MonitorElement* meHitEvsZ_;
  MonitorElement* meHitTvsPhi_;
  MonitorElement* meHitTvsEta_;
  MonitorElement* meHitTvsZ_;
};

// ------------ constructor and destructor --------------
BtlSimHitsValidation::BtlSimHitsValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      hitMinEnergy_(iConfig.getParameter<double>("hitMinimumEnergy")) {
  btlSimHitsToken_ = consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("inputTag"));
}

BtlSimHitsValidation::~BtlSimHitsValidation() {}

// ------------ method called for each event  ------------
void BtlSimHitsValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace geant_units::operators;

  edm::ESHandle<MTDGeometry> geometryHandle;
  iSetup.get<MTDDigiGeometryRecord>().get(geometryHandle);
  const MTDGeometry* geom = geometryHandle.product();

  edm::ESHandle<MTDTopology> topologyHandle;
  iSetup.get<MTDTopologyRcd>().get(topologyHandle);
  const MTDTopology* topology = topologyHandle.product();

  auto btlSimHitsHandle = makeValid(iEvent.getHandle(btlSimHitsToken_));

  std::unordered_map<uint32_t, MTDHit> m_btlHits;
  std::unordered_map<uint32_t, std::set<int> > m_btlTrkPerCell;

  // --- Loop over the BLT SIM hits
  for (auto const& simHit : *btlSimHitsHandle) {
    // --- Use only hits compatible with the in-time bunch-crossing
    if (simHit.tof() < 0 || simHit.tof() > 25.)
      continue;

    DetId id = simHit.detUnitId();

    m_btlTrkPerCell[id.rawId()].insert(simHit.trackId());

    auto simHitIt = m_btlHits.emplace(id.rawId(), MTDHit()).first;

    // --- Accumulate the energy (in MeV) of SIM hits in the same detector cell
    (simHitIt->second).energy += convertUnitsTo(0.001_MeV, simHit.energyLoss());

    // --- Get the time of the first SIM hit in the cell
    if ((simHitIt->second).time == 0 || simHit.tof() < (simHitIt->second).time) {
      (simHitIt->second).time = simHit.tof();

      auto hit_pos = simHit.entryPoint();
      (simHitIt->second).x = hit_pos.x();
      (simHitIt->second).y = hit_pos.y();
      (simHitIt->second).z = hit_pos.z();
    }

  }  // simHit loop

  // ==============================================================================
  //  Histogram filling
  // ==============================================================================

  meNhits_->Fill(m_btlHits.size());

  for (auto const& hit : m_btlTrkPerCell)
    meNtrkPerCell_->Fill((hit.second).size());

  for (auto const& hit : m_btlHits) {
    if ((hit.second).energy < hitMinEnergy_)
      continue;

    // --- Get the SIM hit global position
    BTLDetId detId(hit.first);
    DetId geoId = detId.geographicalId(static_cast<BTLDetId::CrysLayout>(topology->getMTDTopologyMode()));
    const MTDGeomDet* thedet = geom->idToDet(geoId);
    if (thedet == nullptr)
      throw cms::Exception("BtlSimHitsValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                   << detId.rawId() << ") is invalid!" << std::dec << std::endl;
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

    Local3DPoint local_point(
        convertMmToCm((hit.second).x), convertMmToCm((hit.second).y), convertMmToCm((hit.second).z));

    local_point = topo.pixelToModuleLocalPoint(local_point, detId.row(topo.nrows()), detId.column(topo.nrows()));
    const auto& global_point = thedet->toGlobal(local_point);

    // --- Fill the histograms
    meHitEnergy_->Fill((hit.second).energy);
    meHitTime_->Fill((hit.second).time);

    meHitXlocal_->Fill((hit.second).x);
    meHitYlocal_->Fill((hit.second).y);
    meHitZlocal_->Fill((hit.second).z);

    meOccupancy_->Fill(global_point.z(), global_point.phi());

    meHitX_->Fill(global_point.x());
    meHitY_->Fill(global_point.y());
    meHitZ_->Fill(global_point.z());
    meHitPhi_->Fill(global_point.phi());
    meHitEta_->Fill(global_point.eta());

    meHitTvsE_->Fill((hit.second).energy, (hit.second).time);
    meHitEvsPhi_->Fill(global_point.phi(), (hit.second).energy);
    meHitEvsEta_->Fill(global_point.eta(), (hit.second).energy);
    meHitEvsZ_->Fill(global_point.z(), (hit.second).energy);
    meHitTvsPhi_->Fill(global_point.phi(), (hit.second).time);
    meHitTvsEta_->Fill(global_point.eta(), (hit.second).time);
    meHitTvsZ_->Fill(global_point.z(), (hit.second).time);

  }  // hit loop
}

// ------------ method for histogram booking ------------
void BtlSimHitsValidation::bookHistograms(DQMStore::IBooker& ibook,
                                          edm::Run const& run,
                                          edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  // --- histograms booking

  meNhits_ = ibook.book1D("BtlNhits", "Number of BTL cells with SIM hits;N_{BTL cells}", 100, 0., 5000.);
  meNtrkPerCell_ = ibook.book1D("BtlNtrkPerCell", "Number of tracks per BTL cell;N_{trk}", 10, 0., 10.);

  meHitEnergy_ = ibook.book1D("BtlHitEnergy", "BTL SIM hits energy;E_{SIM} [MeV]", 100, 0., 20.);
  meHitTime_ = ibook.book1D("BtlHitTime", "BTL SIM hits ToA;ToA_{SIM} [ns]", 100, 0., 25.);

  meHitXlocal_ = ibook.book1D("BtlHitXlocal", "BTL SIM local X;X_{SIM}^{LOC} [mm]", 100, -30., 30.);
  meHitYlocal_ = ibook.book1D("BtlHitYlocal", "BTL SIM local Y;Y_{SIM}^{LOC} [mm]", 100, -1.65, 1.65);
  meHitZlocal_ = ibook.book1D("BtlHitZlocal", "BTL SIM local z;z_{SIM}^{LOC} [mm]", 100, -2., 2.);

  meOccupancy_ = ibook.book2D(
      "BtlOccupancy", "BTL SIM hits occupancy;z_{SIM} [cm];#phi_{SIM} [rad]", 130, -260., 260., 200, -3.15, 3.15);

  meHitX_ = ibook.book1D("BtlHitX", "BTL SIM hits X;X_{SIM} [cm]", 100, -120., 120.);
  meHitY_ = ibook.book1D("BtlHitY", "BTL SIM hits Y;Y_{SIM} [cm]", 100, -120., 120.);
  meHitZ_ = ibook.book1D("BtlHitZ", "BTL SIM hits Z;Z_{SIM} [cm]", 100, -260., 260.);
  meHitPhi_ = ibook.book1D("BtlHitPhi", "BTL SIM hits #phi;#phi_{SIM} [rad]", 200, -3.15, 3.15);
  meHitEta_ = ibook.book1D("BtlHitEta", "BTL SIM hits #eta;#eta_{SIM}", 100, -1.55, 1.55);

  meHitTvsE_ =
      ibook.bookProfile("BtlHitTvsE", "BTL SIM time vs energy;E_{SIM} [MeV];T_{SIM} [ns]", 50, 0., 20., 0., 100.);
  meHitEvsPhi_ = ibook.bookProfile(
      "BtlHitEvsPhi", "BTL SIM energy vs #phi;#phi_{SIM} [rad];E_{SIM} [MeV]", 50, -3.15, 3.15, 0., 100.);
  meHitEvsEta_ =
      ibook.bookProfile("BtlHitEvsEta", "BTL SIM energy vs #eta;#eta_{SIM};E_{SIM} [MeV]", 50, -1.55, 1.55, 0., 100.);
  meHitEvsZ_ =
      ibook.bookProfile("BtlHitEvsZ", "BTL SIM energy vs Z;Z_{SIM} [cm];E_{SIM} [MeV]", 50, -260., 260., 0., 100.);
  meHitTvsPhi_ = ibook.bookProfile(
      "BtlHitTvsPhi", "BTL SIM time vs #phi;#phi_{SIM} [rad];T_{SIM} [ns]", 50, -3.15, 3.15, 0., 100.);
  meHitTvsEta_ =
      ibook.bookProfile("BtlHitTvsEta", "BTL SIM time vs #eta;#eta_{SIM};T_{SIM} [ns]", 50, -1.55, 1.55, 0., 100.);
  meHitTvsZ_ =
      ibook.bookProfile("BtlHitTvsZ", "BTL SIM time vs Z;Z_{SIM} [cm];T_{SIM} [ns]", 50, -260., 260., 0., 100.);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BtlSimHitsValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/BTL/SimHits");
  desc.add<edm::InputTag>("inputTag", edm::InputTag("g4SimHits", "FastTimerHitsBarrel"));
  desc.add<double>("hitMinimumEnergy", 1.);  // [MeV]

  descriptions.add("btlSimHits", desc);
}

DEFINE_FWK_MODULE(BtlSimHitsValidation);
