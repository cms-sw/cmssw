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
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

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

  edm::EDGetTokenT<FTLRecHitCollection> btlRecHitsToken_;

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
};

// ------------ constructor and destructor --------------
BtlRecHitsValidation::BtlRecHitsValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")) {
  btlRecHitsToken_ = consumes<FTLRecHitCollection>(iConfig.getParameter<edm::InputTag>("inputTag"));
}

BtlRecHitsValidation::~BtlRecHitsValidation() {}

// ------------ method called for each event  ------------
void BtlRecHitsValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::ESHandle<MTDGeometry> geometryHandle;
  iSetup.get<MTDDigiGeometryRecord>().get(geometryHandle);
  const MTDGeometry* geom = geometryHandle.product();

  edm::ESHandle<MTDTopology> topologyHandle;
  iSetup.get<MTDTopologyRcd>().get(topologyHandle);
  const MTDTopology* topology = topologyHandle.product();

  auto btlRecHitsHandle = makeValid(iEvent.getHandle(btlRecHitsToken_));

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

    n_reco_btl++;

  }  // recHit loop

  meNhits_->Fill(n_reco_btl);
}

// ------------ method for histogram booking ------------
void BtlRecHitsValidation::bookHistograms(DQMStore::IBooker& ibook,
                                          edm::Run const& run,
                                          edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  // --- histograms booking

  meNhits_ = ibook.book1D("BtlNhits", "Number of BTL RECO hits;N_{RECO}", 100, 0., 5000.);

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
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BtlRecHitsValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/BTL/RecHits");
  desc.add<edm::InputTag>("inputTag", edm::InputTag("mtdRecHits", "FTLBarrel"));

  descriptions.add("btlRecHits", desc);
}

DEFINE_FWK_MODULE(BtlRecHitsValidation);
