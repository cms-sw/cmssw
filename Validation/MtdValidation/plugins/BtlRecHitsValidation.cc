// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      BtlRecHitsValidation
//
/**\class BtlRecHitsValidation BtlRecHitsValidation.cc Validation/MtdValidation/plugins/BtlRecHitsValidation.cc

 Description: BTL RECO hits validation

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
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

struct MTDHit {
  float energy;
  float time;
  float x_local;
  float y_local;
  float z_local;
};

class BtlRecHitsValidation : public DQMEDAnalyzer {
public:
  explicit BtlRecHitsValidation(const edm::ParameterSet&);
  ~BtlRecHitsValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ------------ member data ------------

  const std::string folder_;
  const float hitMinEnergy_;

  edm::EDGetTokenT<FTLRecHitCollection> btlRecHitsToken_;
  edm::EDGetTokenT<CrossingFrame<PSimHit> > btlSimHitsToken_;

  // --- histograms declaration

  MonitorElement* meNhits_;

  MonitorElement* meHitEnergy_;
  MonitorElement* meHitTime_;

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

  MonitorElement* meTimeRes_;
  MonitorElement* meEnergyRes_;
  MonitorElement* meTresvsE_;
  MonitorElement* meEresvsE_;
};

// ------------ constructor and destructor --------------
BtlRecHitsValidation::BtlRecHitsValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      hitMinEnergy_(iConfig.getParameter<double>("hitMinimumEnergy")) {
  btlRecHitsToken_ = consumes<FTLRecHitCollection>(iConfig.getParameter<edm::InputTag>("inputTag"));
  btlSimHitsToken_ = consumes<CrossingFrame<PSimHit> >(iConfig.getParameter<edm::InputTag>("simHitsTag"));
}

BtlRecHitsValidation::~BtlRecHitsValidation() {}

// ------------ method called for each event  ------------
void BtlRecHitsValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace geant_units::operators;

  edm::ESHandle<MTDGeometry> geometryHandle;
  iSetup.get<MTDDigiGeometryRecord>().get(geometryHandle);
  const MTDGeometry* geom = geometryHandle.product();

  edm::ESHandle<MTDTopology> topologyHandle;
  iSetup.get<MTDTopologyRcd>().get(topologyHandle);
  const MTDTopology* topology = topologyHandle.product();

  auto btlRecHitsHandle = makeValid(iEvent.getHandle(btlRecHitsToken_));
  auto btlSimHitsHandle = makeValid(iEvent.getHandle(btlSimHitsToken_));
  MixCollection<PSimHit> btlSimHits(btlSimHitsHandle.product());

  // --- Loop over the BLT SIM hits
  std::unordered_map<uint32_t, MTDHit> m_btlSimHits;
  for (auto const& simHit : btlSimHits) {
    // --- Use only hits compatible with the in-time bunch-crossing
    if (simHit.tof() < 0 || simHit.tof() > 25.)
      continue;

    DetId id = simHit.detUnitId();

    auto simHitIt = m_btlSimHits.emplace(id.rawId(), MTDHit()).first;

    // --- Accumulate the energy (in MeV) of SIM hits in the same detector cell
    (simHitIt->second).energy += convertUnitsTo(0.001_MeV, simHit.energyLoss());

    // --- Get the time of the first SIM hit in the cell
    if ((simHitIt->second).time == 0 || simHit.tof() < (simHitIt->second).time) {
      (simHitIt->second).time = simHit.tof();

      auto hit_pos = simHit.entryPoint();
      (simHitIt->second).x_local = hit_pos.x();
      (simHitIt->second).y_local = hit_pos.y();
      (simHitIt->second).z_local = hit_pos.z();
    }

  }  // simHit loop

  // --- Loop over the BLT RECO hits

  unsigned int n_reco_btl = 0;

  for (const auto& recHit : *btlRecHitsHandle) {
    BTLDetId detId = recHit.id();
    DetId geoId = detId.geographicalId(static_cast<BTLDetId::CrysLayout>(topology->getMTDTopologyMode()));
    const MTDGeomDet* thedet = geom->idToDet(geoId);
    if (thedet == nullptr)
      throw cms::Exception("BtlRecHitsValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                   << detId.rawId() << ") is invalid!" << std::dec << std::endl;
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

    Local3DPoint local_point(0., 0., 0.);
    local_point = topo.pixelToModuleLocalPoint(local_point, detId.row(topo.nrows()), detId.column(topo.nrows()));
    const auto& global_point = thedet->toGlobal(local_point);

    meHitEnergy_->Fill(recHit.energy());
    meHitTime_->Fill(recHit.time());

    meOccupancy_->Fill(global_point.z(), global_point.phi());

    meHitX_->Fill(global_point.x());
    meHitY_->Fill(global_point.y());
    meHitZ_->Fill(global_point.z());
    meHitPhi_->Fill(global_point.phi());
    meHitEta_->Fill(global_point.eta());

    meHitTvsE_->Fill(recHit.energy(), recHit.time());
    meHitEvsPhi_->Fill(global_point.phi(), recHit.energy());
    meHitEvsEta_->Fill(global_point.eta(), recHit.energy());
    meHitEvsZ_->Fill(global_point.z(), recHit.energy());
    meHitTvsPhi_->Fill(global_point.phi(), recHit.time());
    meHitTvsEta_->Fill(global_point.eta(), recHit.time());
    meHitTvsZ_->Fill(global_point.z(), recHit.time());

    // Resolution histograms
    if (m_btlSimHits.count(detId.rawId()) == 1 && m_btlSimHits[detId.rawId()].energy > hitMinEnergy_) {
      float time_res = recHit.time() - m_btlSimHits[detId.rawId()].time;
      float energy_res = recHit.energy() - m_btlSimHits[detId.rawId()].energy;

      meTimeRes_->Fill(time_res);
      meEnergyRes_->Fill(energy_res);

      meTresvsE_->Fill(m_btlSimHits[detId.rawId()].energy, time_res);
      meEresvsE_->Fill(m_btlSimHits[detId.rawId()].energy, energy_res);
    }

    n_reco_btl++;

  }  // recHit loop

  if (n_reco_btl > 0)
    meNhits_->Fill(log10(n_reco_btl));
}

// ------------ method for histogram booking ------------
void BtlRecHitsValidation::bookHistograms(DQMStore::IBooker& ibook,
                                          edm::Run const& run,
                                          edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  // --- histograms booking

  meNhits_ = ibook.book1D("BtlNhits", "Number of BTL RECO hits;log_{10}(N_{RECO})", 100, 0., 5.25);

  meHitEnergy_ = ibook.book1D("BtlHitEnergy", "BTL RECO hits energy;E_{RECO} [MeV]", 100, 0., 20.);
  meHitTime_ = ibook.book1D("BtlHitTime", "BTL RECO hits ToA;ToA_{RECO} [ns]", 100, 0., 25.);

  meOccupancy_ = ibook.book2D(
      "BtlOccupancy", "BTL RECO hits occupancy;Z_{RECO} [cm]; #phi_{RECO} [rad]", 65, -260., 260., 126, -3.15, 3.15);

  meHitX_ = ibook.book1D("BtlHitX", "BTL RECO hits X;X_{RECO} [cm]", 60, -120., 120.);
  meHitY_ = ibook.book1D("BtlHitY", "BTL RECO hits Y;Y_{RECO} [cm]", 60, -120., 120.);
  meHitZ_ = ibook.book1D("BtlHitZ", "BTL RECO hits Z;Z_{RECO} [cm]", 100, -260., 260.);
  meHitPhi_ = ibook.book1D("BtlHitPhi", "BTL RECO hits #phi;#phi_{RECO} [rad]", 126, -3.15, 3.15);
  meHitEta_ = ibook.book1D("BtlHitEta", "BTL RECO hits #eta;#eta_{RECO}", 100, -1.55, 1.55);

  meHitTvsE_ =
      ibook.bookProfile("BtlHitTvsE", "BTL RECO ToA vs energy;E_{RECO} [MeV];ToA_{RECO} [ns]", 50, 0., 20., 0., 100.);
  meHitEvsPhi_ = ibook.bookProfile(
      "BtlHitEvsPhi", "BTL RECO energy vs #phi;#phi_{RECO} [rad];E_{RECO} [MeV]", 50, -3.15, 3.15, 0., 100.);
  meHitEvsEta_ = ibook.bookProfile(
      "BtlHitEvsEta", "BTL RECO energy vs #eta;#eta_{RECO};E_{RECO} [MeV]", 50, -1.55, 1.55, 0., 100.);
  meHitEvsZ_ =
      ibook.bookProfile("BtlHitEvsZ", "BTL RECO energy vs Z;Z_{RECO} [cm];E_{RECO} [MeV]", 50, -260., 260., 0., 100.);
  meHitTvsPhi_ = ibook.bookProfile(
      "BtlHitTvsPhi", "BTL RECO ToA vs #phi;#phi_{RECO} [rad];ToA_{RECO} [ns]", 50, -3.15, 3.15, 0., 100.);
  meHitTvsEta_ =
      ibook.bookProfile("BtlHitTvsEta", "BTL RECO ToA vs #eta;#eta_{RECO};ToA_{RECO} [ns]", 50, -1.6, 1.6, 0., 100.);
  meHitTvsZ_ =
      ibook.bookProfile("BtlHitTvsZ", "BTL RECO ToA vs Z;Z_{RECO} [cm];ToA_{RECO} [ns]", 50, -260., 260., 0., 100.);

  meTimeRes_ = ibook.book1D("BtlTimeRes", "BTL time resolution;T_{RECO} - T_{SIM} [ns]", 100, -0.5, 0.5);
  meEnergyRes_ = ibook.book1D("BtlEnergyRes", "BTL energy resolution;E_{RECO} - E_{SIM} [MeV]", 100, -0.5, 0.5);

  meTresvsE_ = ibook.bookProfile(
      "BtlTresvsE", "BTL time resolution vs E;E_{SIM} [MeV];T_{RECO}-T_{SIM} [ns]", 50, 0., 20., 0., 100.);
  meEresvsE_ = ibook.bookProfile(
      "BtlEresvsE", "BTL energy resolution vs E;E_{SIM} [MeV];E_{RECO}-E_{SIM} [MeV]", 50, 0., 20., 0., 100.);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BtlRecHitsValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/BTL/RecHits");
  desc.add<edm::InputTag>("inputTag", edm::InputTag("mtdRecHits", "FTLBarrel"));
  desc.add<edm::InputTag>("simHitsTag", edm::InputTag("mix", "g4SimHitsFastTimerHitsBarrel"));
  desc.add<double>("hitMinimumEnergy", 1.);  // [MeV]

  descriptions.add("btlRecHits", desc);
}

DEFINE_FWK_MODULE(BtlRecHitsValidation);
