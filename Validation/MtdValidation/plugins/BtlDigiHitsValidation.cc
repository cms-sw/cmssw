// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      BtlDigiHitsValidation
//
/**\class BtlDigiHitsValidation BtlDigiHitsValidation.cc Validation/MtdValidation/plugins/BtlDigiHitsValidation.cc

 Description: BTL DIGI hits validation

 Implementation:
     [Notes on implementation]
*/

#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/ValidHandle.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

class BtlDigiHitsValidation : public DQMEDAnalyzer {
public:
  explicit BtlDigiHitsValidation(const edm::ParameterSet&);
  ~BtlDigiHitsValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ------------ member data ------------

  const std::string folder_;

  edm::EDGetTokenT<BTLDigiCollection> btlDigiHitsToken_;

  // --- histograms declaration

  MonitorElement* meNhits_[2];

  MonitorElement* meHitCharge_[2];
  MonitorElement* meHitTime_[2];

  MonitorElement* meOccupancy_[2];

  MonitorElement* meHitX_[2];
  MonitorElement* meHitY_[2];
  MonitorElement* meHitZ_[2];
  MonitorElement* meHitPhi_[2];
  MonitorElement* meHitEta_[2];

  MonitorElement* meHitTvsQ_[2];
  MonitorElement* meHitQvsPhi_[2];
  MonitorElement* meHitQvsEta_[2];
  MonitorElement* meHitQvsZ_[2];
  MonitorElement* meHitTvsPhi_[2];
  MonitorElement* meHitTvsEta_[2];
  MonitorElement* meHitTvsZ_[2];
};

// ------------ constructor and destructor --------------
BtlDigiHitsValidation::BtlDigiHitsValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")) {
  btlDigiHitsToken_ = consumes<BTLDigiCollection>(iConfig.getParameter<edm::InputTag>("inputTag"));
}

BtlDigiHitsValidation::~BtlDigiHitsValidation() {}

// ------------ method called for each event  ------------
void BtlDigiHitsValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::ESHandle<MTDGeometry> geometryHandle;
  iSetup.get<MTDDigiGeometryRecord>().get(geometryHandle);
  const MTDGeometry* geom = geometryHandle.product();

  edm::ESHandle<MTDTopology> topologyHandle;
  iSetup.get<MTDTopologyRcd>().get(topologyHandle);
  const MTDTopology* topology = topologyHandle.product();

  auto btlDigiHitsHandle = makeValid(iEvent.getHandle(btlDigiHitsToken_));

  // --- Loop over the BLT DIGI hits

  unsigned int n_digi_btl[2] = {0, 0};

  for (const auto& dataFrame : *btlDigiHitsHandle) {
    BTLDetId detId = dataFrame.id();
    DetId geoId = detId.geographicalId(static_cast<BTLDetId::CrysLayout>(topology->getMTDTopologyMode()));
    const MTDGeomDet* thedet = geom->idToDet(geoId);
    if (thedet == nullptr)
      throw cms::Exception("BtlDigiHitsValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                    << detId.rawId() << ") is invalid!" << std::dec << std::endl;
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

    Local3DPoint local_point(0., 0., 0.);
    local_point = topo.pixelToModuleLocalPoint(local_point, detId.row(topo.nrows()), detId.column(topo.nrows()));
    const auto& global_point = thedet->toGlobal(local_point);

    const auto& sample_L = dataFrame.sample(0);
    const auto& sample_R = dataFrame.sample(1);

    uint32_t adc[2] = {sample_L.data(), sample_R.data()};
    uint32_t tdc[2] = {sample_L.toa(), sample_R.toa()};

    for (int iside = 0; iside < 2; ++iside) {
      if (adc[iside] == 0)
        continue;

      meHitCharge_[iside]->Fill(adc[iside]);
      meHitTime_[iside]->Fill(tdc[iside]);

      meOccupancy_[iside]->Fill(global_point.z(), global_point.phi());

      meHitX_[iside]->Fill(global_point.x());
      meHitY_[iside]->Fill(global_point.y());
      meHitZ_[iside]->Fill(global_point.z());
      meHitPhi_[iside]->Fill(global_point.phi());
      meHitEta_[iside]->Fill(global_point.eta());

      meHitTvsQ_[iside]->Fill(adc[iside], tdc[iside]);
      meHitQvsPhi_[iside]->Fill(global_point.phi(), adc[iside]);
      meHitQvsEta_[iside]->Fill(global_point.eta(), adc[iside]);
      meHitQvsZ_[iside]->Fill(global_point.z(), adc[iside]);
      meHitTvsPhi_[iside]->Fill(global_point.phi(), tdc[iside]);
      meHitTvsEta_[iside]->Fill(global_point.eta(), tdc[iside]);
      meHitTvsZ_[iside]->Fill(global_point.z(), tdc[iside]);

      n_digi_btl[iside]++;

    }  // iside loop

  }  // dataFrame loop

  meNhits_[0]->Fill(n_digi_btl[0]);
  meNhits_[1]->Fill(n_digi_btl[1]);
}

// ------------ method for histogram booking ------------
void BtlDigiHitsValidation::bookHistograms(DQMStore::IBooker& ibook,
                                           edm::Run const& run,
                                           edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  // --- histograms booking

  meNhits_[0] = ibook.book1D("BtlNhitsL", "Number of BTL DIGI hits (L);N_{DIGI}", 100, 0., 5000.);
  meNhits_[1] = ibook.book1D("BtlNhitsR", "Number of BTL DIGI hits (R);N_{DIGI}", 100, 0., 5000.);

  meHitCharge_[0] = ibook.book1D("BtlHitChargeL", "BTL DIGI hits charge (L);Q_{DIGI} [ADC counts]", 100, 0., 1024.);
  meHitCharge_[1] = ibook.book1D("BtlHitChargeR", "BTL DIGI hits charge (R);Q_{DIGI} [ADC counts]", 100, 0., 1024.);
  meHitTime_[0] = ibook.book1D("BtlHitTimeL", "BTL DIGI hits ToA (L);ToA_{DIGI} [TDC counts]", 100, 0., 1024.);
  meHitTime_[1] = ibook.book1D("BtlHitTimeR", "BTL DIGI hits ToA (R);ToA_{DIGI} [TDC counts]", 100, 0., 1024.);

  meOccupancy_[0] = ibook.book2D("BtlOccupancyL",
                                 "BTL DIGI hits occupancy (L);Z_{DIGI} [cm]; #phi_{DIGI} [rad]",
                                 65,
                                 -260.,
                                 260.,
                                 126,
                                 -3.15,
                                 3.15);
  meOccupancy_[1] = ibook.book2D("BtlOccupancyR",
                                 "BTL DIGI hits occupancy (R);Z_{DIGI} [cm]; #phi_{DIGI} [rad]",
                                 65,
                                 -260.,
                                 260.,
                                 126,
                                 -3.15,
                                 3.15);

  meHitX_[0] = ibook.book1D("BtlHitXL", "BTL DIGI hits X (L);X_{DIGI} [cm]", 60, -120., 120.);
  meHitX_[1] = ibook.book1D("BtlHitXR", "BTL DIGI hits X (R);X_{DIGI} [cm]", 60, -120., 120.);
  meHitY_[0] = ibook.book1D("BtlHitYL", "BTL DIGI hits Y (L);Y_{DIGI} [cm]", 60, -120., 120.);
  meHitY_[1] = ibook.book1D("BtlHitYR", "BTL DIGI hits Y (R);Y_{DIGI} [cm]", 60, -120., 120.);
  meHitZ_[0] = ibook.book1D("BtlHitZL", "BTL DIGI hits Z (L);Z_{DIGI} [cm]", 100, -260., 260.);
  meHitZ_[1] = ibook.book1D("BtlHitZR", "BTL DIGI hits Z (R);Z_{DIGI} [cm]", 100, -260., 260.);
  meHitPhi_[0] = ibook.book1D("BtlHitPhiL", "BTL DIGI hits #phi (L);#phi_{DIGI} [rad]", 126, -3.15, 3.15);
  meHitPhi_[1] = ibook.book1D("BtlHitPhiR", "BTL DIGI hits #phi (R);#phi_{DIGI} [rad]", 126, -3.15, 3.15);
  meHitEta_[0] = ibook.book1D("BtlHitEtaL", "BTL DIGI hits #eta (L);#eta_{DIGI}", 100, -1.55, 1.55);
  meHitEta_[1] = ibook.book1D("BtlHitEtaR", "BTL DIGI hits #eta (R);#eta_{DIGI}", 100, -1.55, 1.55);

  meHitTvsQ_[0] = ibook.bookProfile("BtlHitTvsQL",
                                    "BTL DIGI ToA vs charge (L);Q_{DIGI} [ADC counts];ToA_{DIGI} [TDC counts]",
                                    50,
                                    0.,
                                    1024.,
                                    0.,
                                    1024.);
  meHitTvsQ_[1] = ibook.bookProfile("BtlHitTvsQR",
                                    "BTL DIGI ToA vs charge (R);Q_{DIGI} [ADC counts];ToA_{DIGI} [TDC counts]",
                                    50,
                                    0.,
                                    1024.,
                                    0.,
                                    1024.);
  meHitQvsPhi_[0] = ibook.bookProfile("BtlHitQvsPhiL",
                                      "BTL DIGI charge vs #phi (L);#phi_{DIGI} [rad];Q_{DIGI} [ADC counts]",
                                      50,
                                      -3.15,
                                      3.15,
                                      0.,
                                      1024.);
  meHitQvsPhi_[1] = ibook.bookProfile("BtlHitQvsPhiR",
                                      "BTL DIGI charge vs #phi (R);#phi_{DIGI} [rad];Q_{DIGI} [ADC counts]",
                                      50,
                                      -3.15,
                                      3.15,
                                      0.,
                                      1024.);
  meHitQvsEta_[0] = ibook.bookProfile(
      "BtlHitQvsEtaL", "BTL DIGI charge vs #eta (L);#eta_{DIGI};Q_{DIGI} [ADC counts]", 50, -1.55, 1.55, 0., 1024.);
  meHitQvsEta_[1] = ibook.bookProfile(
      "BtlHitQvsEtaR", "BTL DIGI charge vs #eta (R);#eta_{DIGI};Q_{DIGI} [ADC counts]", 50, -1.55, 1.55, 0., 1024.);
  meHitQvsZ_[0] = ibook.bookProfile(
      "BtlHitQvsZL", "BTL DIGI charge vs Z (L);Z_{DIGI} [cm];Q_{DIGI} [ADC counts]", 50, -260., 260., 0., 1024.);
  meHitQvsZ_[1] = ibook.bookProfile(
      "BtlHitQvsZR", "BTL DIGI charge vs Z (R);Z_{DIGI} [cm];Q_{DIGI} [ADC counts]", 50, -260., 260., 0., 1024.);
  meHitTvsPhi_[0] = ibook.bookProfile(
      "BtlHitTvsPhiL", "BTL DIGI ToA vs #phi (L);#phi_{DIGI} [rad];ToA_{DIGI} [TDC counts]", 50, -3.15, 3.15, 0., 1024.);
  meHitTvsPhi_[1] = ibook.bookProfile(
      "BtlHitTvsPhiR", "BTL DIGI ToA vs #phi (R);#phi_{DIGI} [rad];ToA_{DIGI} [TDC counts]", 50, -3.15, 3.15, 0., 1024.);
  meHitTvsEta_[0] = ibook.bookProfile(
      "BtlHitTvsEtaL", "BTL DIGI ToA vs #eta (L);#eta_{DIGI};ToA_{DIGI} [TDC counts]", 50, -1.55, 1.55, 0., 1024.);
  meHitTvsEta_[1] = ibook.bookProfile(
      "BtlHitTvsEtaR", "BTL DIGI ToA vs #eta (R);#eta_{DIGI};ToA_{DIGI} [TDC counts]", 50, -1.55, 1.55, 0., 1024.);
  meHitTvsZ_[0] = ibook.bookProfile(
      "BtlHitTvsZL", "BTL DIGI ToA vs Z (L);Z_{DIGI} [cm];ToA_{DIGI} [TDC counts]", 50, -260., 260., 0., 1024.);
  meHitTvsZ_[1] = ibook.bookProfile(
      "BtlHitTvsZR", "BTL DIGI ToA vs Z (R);Z_{DIGI} [cm];ToA_{DIGI} [TDC counts]", 50, -260., 260., 0., 1024.);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BtlDigiHitsValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/BTL/DigiHits");
  desc.add<edm::InputTag>("inputTag", edm::InputTag("mix", "FTLBarrel"));

  descriptions.add("btlDigiHitsDefault", desc);
}

DEFINE_FWK_MODULE(BtlDigiHitsValidation);
