// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      BtlDigiSoAHitsValidation
//
/**\class BtlDigiSoAHitsValidation BtlDigiSoAHitsValidation.cc Validation/MtdValidation/plugins/BtlDigiSoAHitsValidation.cc

 Description: BTL DIGI hits validation

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
#include "DataFormats/FTLDigiSoA/interface/BTLDigiHostCollection.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"

#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

class BtlDigiSoAHitsValidation : public DQMEDAnalyzer {
public:
  explicit BtlDigiSoAHitsValidation(const edm::ParameterSet&);
  ~BtlDigiSoAHitsValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ------------ member data ------------

  const std::string folder_;
  const bool optionalPlots_;

  edm::EDGetTokenT<btldigi::BTLDigiHostCollection> btlDigiHitsToken_;

  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;

  // --- histograms declaration

  MonitorElement* meNhits_[2];

  MonitorElement* meHitCharge_[2];
  MonitorElement* meHitT1coarse_[2];
  MonitorElement* meHitT2coarse_[2];
  MonitorElement* meHitT1fine_[2];
  MonitorElement* meHitT2fine_[2];

  MonitorElement* meOccupancy_[2];

  //local position monitoring
  MonitorElement* meLocalOccupancy_[2];
  MonitorElement* meHitXlocal_[2];
  MonitorElement* meHitYlocal_[2];
  MonitorElement* meHitZlocal_[2];

  MonitorElement* meHitX_[2];
  MonitorElement* meHitY_[2];
  MonitorElement* meHitZ_[2];
  MonitorElement* meHitPhi_[2];
  MonitorElement* meHitEta_[2];

  MonitorElement* meHitT1coarseVsQ_[2];
  MonitorElement* meHitT2coarseVsQ_[2];
  MonitorElement* meHitT1fineVsQ_[2];
  MonitorElement* meHitT2fineVsQ_[2];
  MonitorElement* meHitQvsPhi_[2];
  MonitorElement* meHitQvsEta_[2];
  MonitorElement* meHitQvsZ_[2];
  MonitorElement* meHitT1coarseVsPhi_[2];
  MonitorElement* meHitT2coarseVsPhi_[2];
  MonitorElement* meHitT1fineVsPhi_[2];
  MonitorElement* meHitT2fineVsPhi_[2];
  MonitorElement* meHitT1coarseVsEta_[2];
  MonitorElement* meHitT2coarseVsEta_[2];
  MonitorElement* meHitT1fineVsEta_[2];
  MonitorElement* meHitT2fineVsEta_[2];
  MonitorElement* meHitT1coarseVsZ_[2];
  MonitorElement* meHitT2coarseVsZ_[2];
  MonitorElement* meHitT1fineVsZ_[2];
  MonitorElement* meHitT2fineVsZ_[2];
};

// ------------ constructor and destructor --------------
BtlDigiSoAHitsValidation::BtlDigiSoAHitsValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      optionalPlots_(iConfig.getParameter<bool>("optionalPlots")) {
  btlDigiHitsToken_ = consumes<btldigi::BTLDigiHostCollection>(iConfig.getParameter<edm::InputTag>("inputTag"));
  mtdgeoToken_ = esConsumes<MTDGeometry, MTDDigiGeometryRecord>();
  mtdtopoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>();
}

BtlDigiSoAHitsValidation::~BtlDigiSoAHitsValidation() {}

// ------------ method called for each event  ------------
void BtlDigiSoAHitsValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  auto geometryHandle = iSetup.getTransientHandle(mtdgeoToken_);
  const MTDGeometry* geom = geometryHandle.product();

  auto topologyHandle = iSetup.getTransientHandle(mtdtopoToken_);
  const MTDTopology* topology = topologyHandle.product();

  auto btlDigiHitsHandle = makeValid(iEvent.getHandle(btlDigiHitsToken_));

  // --- Loop over the BTL DIGI hits

  unsigned int n_digi_btl[2] = {0, 0};
  const auto btlDigiView = btlDigiHitsHandle->view();
  for (int i = 0; i < btlDigiView.metadata().size(); i++) {
    BTLDetId detId = btldigi::rawId(btlDigiView, i);
    DetId geoId = detId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode()));
    const MTDGeomDet* thedet = geom->idToDet(geoId);
    if (thedet == nullptr)
      throw cms::Exception("BtlDigiSoAHitsValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                       << detId.rawId() << ") is invalid!" << std::dec << std::endl;
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

    Local3DPoint local_point(0., 0., 0.);
    local_point = topo.pixelToModuleLocalPoint(local_point, detId.row(topo.nrows()), detId.column(topo.nrows()));
    const auto& global_point = thedet->toGlobal(local_point);

    uint32_t adc[2] = {btldigi::chargeL(btlDigiView, i), btldigi::chargeR(btlDigiView, i)};
    uint32_t T1coarse[2] = {btldigi::t1CoarseL(btlDigiView, i), btldigi::t1CoarseR(btlDigiView, i)};
    uint32_t T2coarse[2] = {btldigi::t2CoarseL(btlDigiView, i), btldigi::t2CoarseR(btlDigiView, i)};
    uint32_t T1fine[2] = {btldigi::t1FineL(btlDigiView, i), btldigi::t1FineR(btlDigiView, i)};
    uint32_t T2fine[2] = {btldigi::t2FineL(btlDigiView, i), btldigi::t2FineR(btlDigiView, i)};

    for (int iside = 0; iside < 2; ++iside) {
      if (adc[iside] == 0)
        continue;

      meHitCharge_[iside]->Fill(adc[iside]);
      meHitT1coarse_[iside]->Fill(T1coarse[iside]);
      meHitT2coarse_[iside]->Fill(T2coarse[iside]);
      meHitT1fine_[iside]->Fill(T1fine[iside]);
      meHitT2fine_[iside]->Fill(T2fine[iside]);

      meOccupancy_[iside]->Fill(global_point.z(), global_point.phi());

      if (optionalPlots_) {
        meLocalOccupancy_[iside]->Fill(local_point.x(), local_point.y());
        meHitXlocal_[iside]->Fill(local_point.x());
        meHitYlocal_[iside]->Fill(local_point.y());
        meHitZlocal_[iside]->Fill(local_point.z());
      }

      meHitX_[iside]->Fill(global_point.x());
      meHitY_[iside]->Fill(global_point.y());
      meHitZ_[iside]->Fill(global_point.z());
      meHitPhi_[iside]->Fill(global_point.phi());
      meHitEta_[iside]->Fill(global_point.eta());

      meHitT1coarseVsQ_[iside]->Fill(adc[iside], T1coarse[iside]);
      meHitT2coarseVsQ_[iside]->Fill(adc[iside], T2coarse[iside]);
      meHitT1fineVsQ_[iside]->Fill(adc[iside], T1fine[iside]);
      meHitT2fineVsQ_[iside]->Fill(adc[iside], T2fine[iside]);

      meHitQvsPhi_[iside]->Fill(global_point.phi(), adc[iside]);
      meHitT1coarseVsPhi_[iside]->Fill(global_point.phi(), T1coarse[iside]);
      meHitT2coarseVsPhi_[iside]->Fill(global_point.phi(), T2coarse[iside]);
      meHitT1fineVsPhi_[iside]->Fill(global_point.phi(), T1fine[iside]);
      meHitT2fineVsPhi_[iside]->Fill(global_point.phi(), T2fine[iside]);

      meHitQvsEta_[iside]->Fill(global_point.eta(), adc[iside]);
      meHitT1coarseVsEta_[iside]->Fill(global_point.eta(), T1coarse[iside]);
      meHitT2coarseVsEta_[iside]->Fill(global_point.eta(), T2coarse[iside]);
      meHitT1fineVsEta_[iside]->Fill(global_point.eta(), T1fine[iside]);
      meHitT2fineVsZ_[iside]->Fill(global_point.z(), T2fine[iside]);

      meHitQvsZ_[iside]->Fill(global_point.z(), adc[iside]);
      meHitT1coarseVsZ_[iside]->Fill(global_point.z(), T1coarse[iside]);
      meHitT2coarseVsZ_[iside]->Fill(global_point.z(), T2coarse[iside]);
      meHitT1fineVsZ_[iside]->Fill(global_point.z(), T1fine[iside]);
      meHitT2fineVsEta_[iside]->Fill(global_point.eta(), T2fine[iside]);

      n_digi_btl[iside]++;

    }  // iside loop

  }  // dataFrame loop

  if (n_digi_btl[0] > 0)
    meNhits_[0]->Fill(log10(n_digi_btl[0]));
  if (n_digi_btl[1] > 0)
    meNhits_[1]->Fill(log10(n_digi_btl[1]));
}

// ------------ method for histogram booking ------------
void BtlDigiSoAHitsValidation::bookHistograms(DQMStore::IBooker& ibook,
                                              edm::Run const& run,
                                              edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);
  // --- histograms booking

  meNhits_[0] = ibook.book1D("BtlNhitsL", "Number of BTL DIGI hits (L);log_{10}(N_{DIGI})", 100, 0., 5.25);
  meNhits_[1] = ibook.book1D("BtlNhitsR", "Number of BTL DIGI hits (R);log_{10}(N_{DIGI})", 100, 0., 5.25);

  meHitCharge_[0] = ibook.book1D("BtlHitChargeL", "BTL DIGI hits charge (L);Q_{DIGI} [ADC counts]", 100, 0., 1024.);
  meHitCharge_[1] = ibook.book1D("BtlHitChargeR", "BTL DIGI hits charge (R);Q_{DIGI} [ADC counts]", 100, 0., 1024.);

  meHitT1coarse_[0] =
      ibook.book1D("BtlHitT1coarseL", "BTL DIGI hits T1 coarse (L);ToA_{DIGI} [# clk cycles]", 100, 0., 1024.);
  meHitT1coarse_[1] =
      ibook.book1D("BtlHitT1coarseR", "BTL DIGI hits T1 coarse (R);ToA_{DIGI} [# clk cycles]", 100, 0., 1024.);
  meHitT2coarse_[0] =
      ibook.book1D("BtlHitT2coarseL", "BTL DIGI hits T2 coarse (L);ToA_{DIGI} [# clk cycles]", 100, 0., 1024.);
  meHitT2coarse_[1] =
      ibook.book1D("BtlHitT2coarseR", "BTL DIGI hits T2 coarse (R);ToA_{DIGI} [# clk cycles]", 100, 0., 1024.);

  meHitT1fine_[0] = ibook.book1D("BtlHitT1fineL", "BTL DIGI hits T1 fine (L);ToA_{DIGI} [TDC counts]", 100, 0., 1024.);
  meHitT1fine_[1] = ibook.book1D("BtlHitT1fineR", "BTL DIGI hits T1 fine (R);ToA_{DIGI} [TDC counts]", 100, 0., 1024.);
  meHitT2fine_[0] = ibook.book1D("BtlHitT2fineL", "BTL DIGI hits T2 fine (L);ToA_{DIGI} [TDC counts]", 100, 0., 1024.);
  meHitT2fine_[1] = ibook.book1D("BtlHitT2fineR", "BTL DIGI hits T2 fine (R);ToA_{DIGI} [TDC counts]", 100, 0., 1024.);

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
  if (optionalPlots_) {
    meLocalOccupancy_[0] = ibook.book2D("BtlLocalOccupancyL",
                                        "BTL DIGI hits local occupancy (L);X_{DIGI} [cm]; Y_{DIGI} [cm]",
                                        100,
                                        -10.,
                                        10,
                                        60,
                                        -3.,
                                        3.);
    meLocalOccupancy_[1] = ibook.book2D(
        "BtlLocalOccupancyR", "BTL DIGI hits occupancy (R);X_{DIGI} [cm]; Y_{DIGI} [cm]", 100, -10., 10., 60, -3., 3.);
    meHitXlocal_[0] = ibook.book1D("BtlHitXlocalL", "BTL DIGI local X (L);X_{DIGI}^{LOC} [cm]", 100, -10., 10.);
    meHitXlocal_[1] = ibook.book1D("BtlHitXlocalR", "BTL DIGI local X (R);X_{DIGI}^{LOC} [cm]", 100, -10., 10.);
    meHitYlocal_[0] = ibook.book1D("BtlHitYlocalL", "BTL DIGI local Y (L);Y_{DIGI}^{LOC} [cm]", 60, -3., 3.);
    meHitYlocal_[1] = ibook.book1D("BtlHitYlocalR", "BTL DIGI local Y (R);Y_{DIGI}^{LOC} [cm]", 60, -3., 3.);
    meHitZlocal_[0] = ibook.book1D("BtlHitZlocalL", "BTL DIGI local z (L);z_{DIGI}^{LOC} [cm]", 10, -1, 1);
    meHitZlocal_[1] = ibook.book1D("BtlHitZlocalR", "BTL DIGI local z (R);z_{DIGI}^{LOC} [cm]", 10, -1, 1);
  }

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

  meHitT1coarseVsQ_[0] =
      ibook.bookProfile("BtlHitT1coarseVsQL",
                        "BTL DIGI T1 coarse vs charge (L);Q_{DIGI} [ADC counts];T1Coarse_{DIGI} [# clk cycles]",
                        50,
                        0.,
                        1024.,
                        0.,
                        1024.);
  meHitT1coarseVsQ_[1] =
      ibook.bookProfile("BtlHitT1coarseVsQR",
                        "BTL DIGI T1 coarse vs charge (R);Q_{DIGI} [ADC counts];T1Coarse_{DIGI} [# clk cycles]",
                        50,
                        0.,
                        1024.,
                        0.,
                        1024.);

  meHitT2coarseVsQ_[0] =
      ibook.bookProfile("BtlHitT2coarseVsQL",
                        "BTL DIGI T2 coarse vs charge (L);Q_{DIGI} [ADC counts];T2Coarse_{DIGI} [# clk cycles]",
                        50,
                        0.,
                        1024.,
                        0.,
                        1024.);
  meHitT2coarseVsQ_[1] =
      ibook.bookProfile("BtlHitT2coarseVsQR",
                        "BTL DIGI T2 coarse vs charge (R);Q_{DIGI} [ADC counts];T2Coarse_{DIGI} [# clk cycles]",
                        50,
                        0.,
                        1024.,
                        0.,
                        1024.);
  meHitT1fineVsQ_[0] =
      ibook.bookProfile("BtlHitT1fineVsQL",
                        "BTL DIGI T1 fine vs charge (L);Q_{DIGI} [ADC counts];T1Fine_{DIGI} [TDC counts]",
                        50,
                        0.,
                        1024.,
                        0.,
                        1024.);
  meHitT1fineVsQ_[1] =
      ibook.bookProfile("BtlHitT1fineVsQR",
                        "BTL DIGI T1 fine vs charge (R);Q_{DIGI} [ADC counts];T1Fine_{DIGI} [TDC counts]",
                        50,
                        0.,
                        1024.,
                        0.,
                        1024.);

  meHitT2fineVsQ_[0] =
      ibook.bookProfile("BtlHitT2fineVsQL",
                        "BTL DIGI T2 fine vs charge (L);Q_{DIGI} [ADC counts];T2Fine_{DIGI} [TDC counts]",
                        50,
                        0.,
                        1024.,
                        0.,
                        1024.);
  meHitT2fineVsQ_[1] =
      ibook.bookProfile("BtlHitT2fineVsQR",
                        "BTL DIGI T2 fine vs charge (R);Q_{DIGI} [ADC counts];T2Fine_{DIGI} [TDC counts]",
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

  meHitT1coarseVsPhi_[0] =
      ibook.bookProfile("BtlHitT1coarseVsPhiL",
                        "BTL DIGI T1 Coarse vs #phi (L);#phi_{DIGI} [rad];T1Coarse_{DIGI} [# clk cycles]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        1024.);
  meHitT1coarseVsPhi_[1] =
      ibook.bookProfile("BtlHitT1coarseVsPhiR",
                        "BTL DIGI T1 Coarse vs #phi (R);#phi_{DIGI} [rad];T1Coarse_{DIGI} [# clk cycles]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        1024.);
  meHitT2coarseVsPhi_[0] =
      ibook.bookProfile("BtlHitT2coarseVsPhiL",
                        "BTL DIGI T2 Coarse vs #phi (L);#phi_{DIGI} [rad];T2Coarse_{DIGI} [# clk cycles]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        1024.);
  meHitT2coarseVsPhi_[1] =
      ibook.bookProfile("BtlHitT2coarseVsPhiR",
                        "BTL DIGI T2 Coarse vs #phi (R);#phi_{DIGI} [rad];T2Coarse_{DIGI} [# clk cycles]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        1024.);
  meHitT1fineVsPhi_[0] = ibook.bookProfile("BtlHitT1fineVsPhiL",
                                           "BTL DIGI T1 Fine vs #phi (L);#phi_{DIGI} [rad];T1Fine_{DIGI} [TDC counts]",
                                           50,
                                           -3.15,
                                           3.15,
                                           0.,
                                           1024.);
  meHitT1fineVsPhi_[1] = ibook.bookProfile("BtlHitT1fineVsPhiR",
                                           "BTL DIGI T1 Fine vs #phi (R);#phi_{DIGI} [rad];T1Fine_{DIGI} [TDC counts]",
                                           50,
                                           -3.15,
                                           3.15,
                                           0.,
                                           1024.);
  meHitT2fineVsPhi_[0] = ibook.bookProfile("BtlHitT2fineVsPhiL",
                                           "BTL DIGI T2 Fine vs #phi (L);#phi_{DIGI} [rad];T2Fine_{DIGI} [TDC counts]",
                                           50,
                                           -3.15,
                                           3.15,
                                           0.,
                                           1024.);
  meHitT2fineVsPhi_[1] = ibook.bookProfile("BtlHitT2fineVsPhiR",
                                           "BTL DIGI T2 Fine vs #phi (R);#phi_{DIGI} [rad];T2Fine_{DIGI} [TDC counts]",
                                           50,
                                           -3.15,
                                           3.15,
                                           0.,
                                           1024.);

  meHitT1coarseVsEta_[0] =
      ibook.bookProfile("BtlHitT1coarseVsEtaL",
                        "BTL DIGI T1 Coarse vs #eta (L);#eta_{DIGI};T1Coarse_{DIGI} [# clk cycles]",
                        50,
                        -1.55,
                        1.55,
                        0.,
                        1024.);
  meHitT1coarseVsEta_[1] =
      ibook.bookProfile("BtlHitT1coarseVsEtaR",
                        "BTL DIGI T1 Coarse vs #eta (R);#eta_{DIGI};T1Coarse_{DIGI} [# clk cycles]",
                        50,
                        -1.55,
                        1.55,
                        0.,
                        1024.);
  meHitT2coarseVsEta_[0] =
      ibook.bookProfile("BtlHitT2coarseVsEtaL",
                        "BTL DIGI T2 Coarse vs #eta (L);#eta_{DIGI};T2Coarse_{DIGI} [# clk cycles]",
                        50,
                        -1.55,
                        1.55,
                        0.,
                        1024.);
  meHitT2coarseVsEta_[1] =
      ibook.bookProfile("BtlHitT2coarseVsEtaR",
                        "BTL DIGI T2 Coarse vs #eta (R);#eta_{DIGI};T2Coarse_{DIGI} [# clk cycles]",
                        50,
                        -1.55,
                        1.55,
                        0.,
                        1024.);
  meHitT1fineVsEta_[0] = ibook.bookProfile("BtlHitT1fineVsEtaL",
                                           "BTL DIGI T1 Fine vs #eta (L);#eta_{DIGI};T1Fine_{DIGI} [TDC counts]",
                                           50,
                                           -1.55,
                                           1.55,
                                           0.,
                                           1024.);
  meHitT1fineVsEta_[1] = ibook.bookProfile("BtlHitT1fineVsEtaR",
                                           "BTL DIGI T1 Fine vs #eta (R);#eta_{DIGI};T1Fine_{DIGI} [TDC counts]",
                                           50,
                                           -1.55,
                                           1.55,
                                           0.,
                                           1024.);
  meHitT2fineVsEta_[0] = ibook.bookProfile("BtlHitT2fineVsEtaL",
                                           "BTL DIGI T2 Fine vs #eta (L);#eta_{DIGI};T2Fine_{DIGI} [TDC counts]",
                                           50,
                                           -1.55,
                                           1.55,
                                           0.,
                                           1024.);
  meHitT2fineVsEta_[1] = ibook.bookProfile("BtlHitT2fineVsEtaR",
                                           "BTL DIGI T2 Fine vs #eta (R);#eta_{DIGI};T2Fine_{DIGI} [TDC counts]",
                                           50,
                                           -1.55,
                                           1.55,
                                           0.,
                                           1024.);

  meHitT1coarseVsZ_[0] = ibook.bookProfile("BtlHitT1coarseVsZL",
                                           "BTL DIGI T1 Coarse vs Z (L);Z_{DIGI} [cm];T1Coarse_{DIGI} [# clk cycles]",
                                           50,
                                           -260.,
                                           260.,
                                           0.,
                                           1024.);
  meHitT1coarseVsZ_[1] = ibook.bookProfile("BtlHitT1coarseVsZR",
                                           "BTL DIGI T1 Coarse vs Z (R);Z_{DIGI} [cm];T1Coarse_{DIGI} [# clk cycles]",
                                           50,
                                           -260.,
                                           260.,
                                           0.,
                                           1024.);
  meHitT2coarseVsZ_[0] = ibook.bookProfile("BtlHitT2coarseVsZL",
                                           "BTL DIGI T2 Coarse vs Z (L);Z_{DIGI} [cm];T2Coarse_{DIGI} [# clk cycles]",
                                           50,
                                           -260.,
                                           260.,
                                           0.,
                                           1024.);
  meHitT2coarseVsZ_[1] = ibook.bookProfile("BtlHitT2coarseVsZR",
                                           "BTL DIGI T2 Coarse vs Z (R);Z_{DIGI} [cm];T2Coarse_{DIGI} [# clk cycles]",
                                           50,
                                           -260.,
                                           260.,
                                           0.,
                                           1024.);
  meHitT1fineVsZ_[0] = ibook.bookProfile("BtlHitT1fineVsZL",
                                         "BTL DIGI T1 Fine vs Z (L);Z_{DIGI} [cm];T1Fine_{DIGI} [TDC counts]",
                                         50,
                                         -260.,
                                         260.,
                                         0.,
                                         1024.);
  meHitT1fineVsZ_[1] = ibook.bookProfile("BtlHitT1fineVsZR",
                                         "BTL DIGI T1 Fine vs Z (R);Z_{DIGI} [cm];T1Fine_{DIGI} [TDC counts]",
                                         50,
                                         -260.,
                                         260.,
                                         0.,
                                         1024.);
  meHitT2fineVsZ_[0] = ibook.bookProfile("BtlHitT2fineVsZL",
                                         "BTL DIGI T2 Fine vs Z (L);Z_{DIGI} [cm];T2Fine_{DIGI} [TDC counts]",
                                         50,
                                         -260.,
                                         260.,
                                         0.,
                                         1024.);
  meHitT2fineVsZ_[1] = ibook.bookProfile("BtlHitT2fineVsZR",
                                         "BTL DIGI T2 Fine vs Z (R);Z_{DIGI} [cm];T2Fine_{DIGI} [TDC counts]",
                                         50,
                                         -260.,
                                         260.,
                                         0.,
                                         1024.);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BtlDigiSoAHitsValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/BTL/DigiHitsSoA");
  desc.add<edm::InputTag>("inputTag", edm::InputTag("mix", "FTLBarrelSoA"));
  desc.add<bool>("optionalPlots", false);

  descriptions.add("btlDigiSoAHitsDefaultValid", desc);
}

DEFINE_FWK_MODULE(BtlDigiSoAHitsValidation);
