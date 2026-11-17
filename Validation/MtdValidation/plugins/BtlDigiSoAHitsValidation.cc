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
    auto digi = btlDigiView[i];
    BTLDetId detId = digi.rawId();
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

    uint32_t adc[2] = {digi.ChargeMinus(), digi.ChargePlus()};
    uint32_t T1coarse[2] = {digi.T1coarseMinus(), digi.T1coarsePlus()};
    uint32_t T2coarse[2] = {digi.T2coarseMinus(), digi.T2coarsePlus()};
    uint32_t T1fine[2] = {digi.T1fineMinus(), digi.T1finePlus()};
    uint32_t T2fine[2] = {digi.T2fineMinus(), digi.T2finePlus()};

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

  meNhits_[0] = ibook.book1D("BtlNhitsMinus", "Number of BTL DIGI hits (- side);log_{10}(N_{DIGI})", 100, 0., 5.25);
  meNhits_[1] = ibook.book1D("BtlNhitsPlus", "Number of BTL DIGI hits (+ side);log_{10}(N_{DIGI})", 100, 0., 5.25);

  meHitCharge_[0] =
      ibook.book1D("BtlHitChargeMinus", "BTL DIGI hits charge (- side);Q_{DIGI} [ADC counts]", 100, 0., 1024.);
  meHitCharge_[1] =
      ibook.book1D("BtlHitChargePlus", "BTL DIGI hits charge (+ side);Q_{DIGI} [ADC counts]", 100, 0., 1024.);

  meHitT1coarse_[0] =
      ibook.book1D("BtlHitT1coarseMinus", "BTL DIGI hits T1 coarse (- side);ToA_{DIGI} [# clk cycles]", 10, 0., 10.);
  meHitT1coarse_[1] =
      ibook.book1D("BtlHitT1coarsePlus", "BTL DIGI hits T1 coarse (+ side);ToA_{DIGI} [# clk cycles]", 10, 0., 10.);
  meHitT2coarse_[0] =
      ibook.book1D("BtlHitT2coarseMinus", "BTL DIGI hits T2 coarse (- side);ToA_{DIGI} [# clk cycles]", 10, 0., 10.);
  meHitT2coarse_[1] =
      ibook.book1D("BtlHitT2coarsePlus", "BTL DIGI hits T2 coarse (+ side);ToA_{DIGI} [# clk cycles]", 10, 0., 10.);

  meHitT1fine_[0] =
      ibook.book1D("BtlHitT1fineMinus", "BTL DIGI hits T1 fine (- side);ToA_{DIGI} [TDC counts]", 100, 0., 1024.);
  meHitT1fine_[1] =
      ibook.book1D("BtlHitT1finePlus", "BTL DIGI hits T1 fine (+ side);ToA_{DIGI} [TDC counts]", 100, 0., 1024.);
  meHitT2fine_[0] =
      ibook.book1D("BtlHitT2fineMinus", "BTL DIGI hits T2 fine (- side);ToA_{DIGI} [TDC counts]", 100, 0., 1024.);
  meHitT2fine_[1] =
      ibook.book1D("BtlHitT2finePlus", "BTL DIGI hits T2 fine (+ side);ToA_{DIGI} [TDC counts]", 100, 0., 1024.);

  meOccupancy_[0] = ibook.book2D("BtlOccupancyMinus",
                                 "BTL DIGI hits occupancy (- side);Z_{DIGI} [cm]; #phi_{DIGI} [rad]",
                                 65,
                                 -260.,
                                 260.,
                                 126,
                                 -3.15,
                                 3.15);
  meOccupancy_[1] = ibook.book2D("BtlOccupancyPlus",
                                 "BTL DIGI hits occupancy (+ side);Z_{DIGI} [cm]; #phi_{DIGI} [rad]",
                                 65,
                                 -260.,
                                 260.,
                                 126,
                                 -3.15,
                                 3.15);
  if (optionalPlots_) {
    meLocalOccupancy_[0] = ibook.book2D("BtlLocalOccupancyMinus",
                                        "BTL DIGI hits local occupancy (- side);X_{DIGI} [cm]; Y_{DIGI} [cm]",
                                        100,
                                        -10.,
                                        10,
                                        60,
                                        -3.,
                                        3.);
    meLocalOccupancy_[1] = ibook.book2D("BtlLocalOccupancyPlus",
                                        "BTL DIGI hits occupancy (+ side);X_{DIGI} [cm]; Y_{DIGI} [cm]",
                                        100,
                                        -10.,
                                        10.,
                                        60,
                                        -3.,
                                        3.);
    meHitXlocal_[0] =
        ibook.book1D("BtlHitXlocalMinus", "BTL DIGI local X (- side);X_{DIGI}^{LOC} [cm]", 100, -10., 10.);
    meHitXlocal_[1] = ibook.book1D("BtlHitXlocalPlus", "BTL DIGI local X (+ side);X_{DIGI}^{LOC} [cm]", 100, -10., 10.);
    meHitYlocal_[0] = ibook.book1D("BtlHitYlocalMinus", "BTL DIGI local Y (- side);Y_{DIGI}^{LOC} [cm]", 60, -3., 3.);
    meHitYlocal_[1] = ibook.book1D("BtlHitYlocalPlus", "BTL DIGI local Y (+ side);Y_{DIGI}^{LOC} [cm]", 60, -3., 3.);
    meHitZlocal_[0] = ibook.book1D("BtlHitZlocalMinus", "BTL DIGI local z (- side);z_{DIGI}^{LOC} [cm]", 10, -1, 1);
    meHitZlocal_[1] = ibook.book1D("BtlHitZlocalPlus", "BTL DIGI local z (+ side);z_{DIGI}^{LOC} [cm]", 10, -1, 1);
  }

  meHitX_[0] = ibook.book1D("BtlHitXMinus", "BTL DIGI hits X (- side);X_{DIGI} [cm]", 60, -120., 120.);
  meHitX_[1] = ibook.book1D("BtlHitXPlus", "BTL DIGI hits X (+ side);X_{DIGI} [cm]", 60, -120., 120.);
  meHitY_[0] = ibook.book1D("BtlHitYMinus", "BTL DIGI hits Y (- side);Y_{DIGI} [cm]", 60, -120., 120.);
  meHitY_[1] = ibook.book1D("BtlHitYPlus", "BTL DIGI hits Y (+ side);Y_{DIGI} [cm]", 60, -120., 120.);
  meHitZ_[0] = ibook.book1D("BtlHitZMinus", "BTL DIGI hits Z (- side);Z_{DIGI} [cm]", 100, -260., 260.);
  meHitZ_[1] = ibook.book1D("BtlHitZPlus", "BTL DIGI hits Z (+ side);Z_{DIGI} [cm]", 100, -260., 260.);
  meHitPhi_[0] = ibook.book1D("BtlHitPhiMinus", "BTL DIGI hits #phi (- side);#phi_{DIGI} [rad]", 126, -3.15, 3.15);
  meHitPhi_[1] = ibook.book1D("BtlHitPhiPlus", "BTL DIGI hits #phi (+ side);#phi_{DIGI} [rad]", 126, -3.15, 3.15);
  meHitEta_[0] = ibook.book1D("BtlHitEtaMinus", "BTL DIGI hits #eta (- side);#eta_{DIGI}", 100, -1.55, 1.55);
  meHitEta_[1] = ibook.book1D("BtlHitEtaPlus", "BTL DIGI hits #eta (+ side);#eta_{DIGI}", 100, -1.55, 1.55);

  meHitT1coarseVsQ_[0] =
      ibook.bookProfile("BtlHitT1coarseVsQMinus",
                        "BTL DIGI T1 coarse vs charge (- side);Q_{DIGI} [ADC counts];T1Coarse_{DIGI} [# clk cycles]",
                        50,
                        0.,
                        1024.,
                        0.,
                        1024.);
  meHitT1coarseVsQ_[1] =
      ibook.bookProfile("BtlHitT1coarseVsQPlus",
                        "BTL DIGI T1 coarse vs charge (+ side);Q_{DIGI} [ADC counts];T1Coarse_{DIGI} [# clk cycles]",
                        50,
                        0.,
                        1024.,
                        0.,
                        1024.);

  meHitT2coarseVsQ_[0] =
      ibook.bookProfile("BtlHitT2coarseVsQMinus",
                        "BTL DIGI T2 coarse vs charge (- side);Q_{DIGI} [ADC counts];T2Coarse_{DIGI} [# clk cycles]",
                        50,
                        0.,
                        1024.,
                        0.,
                        1024.);
  meHitT2coarseVsQ_[1] =
      ibook.bookProfile("BtlHitT2coarseVsQPlus",
                        "BTL DIGI T2 coarse vs charge (+ side);Q_{DIGI} [ADC counts];T2Coarse_{DIGI} [# clk cycles]",
                        50,
                        0.,
                        1024.,
                        0.,
                        1024.);
  meHitT1fineVsQ_[0] =
      ibook.bookProfile("BtlHitT1fineVsQMinus",
                        "BTL DIGI T1 fine vs charge (- side);Q_{DIGI} [ADC counts];T1Fine_{DIGI} [TDC counts]",
                        50,
                        0.,
                        1024.,
                        0.,
                        1024.);
  meHitT1fineVsQ_[1] =
      ibook.bookProfile("BtlHitT1fineVsQPlus",
                        "BTL DIGI T1 fine vs charge (+ side);Q_{DIGI} [ADC counts];T1Fine_{DIGI} [TDC counts]",
                        50,
                        0.,
                        1024.,
                        0.,
                        1024.);

  meHitT2fineVsQ_[0] =
      ibook.bookProfile("BtlHitT2fineVsQMinus",
                        "BTL DIGI T2 fine vs charge (- side);Q_{DIGI} [ADC counts];T2Fine_{DIGI} [TDC counts]",
                        50,
                        0.,
                        1024.,
                        0.,
                        1024.);
  meHitT2fineVsQ_[1] =
      ibook.bookProfile("BtlHitT2fineVsQPlus",
                        "BTL DIGI T2 fine vs charge (+ side);Q_{DIGI} [ADC counts];T2Fine_{DIGI} [TDC counts]",
                        50,
                        0.,
                        1024.,
                        0.,
                        1024.);

  meHitQvsPhi_[0] = ibook.bookProfile("BtlHitQvsPhiMinus",
                                      "BTL DIGI charge vs #phi (- side);#phi_{DIGI} [rad];Q_{DIGI} [ADC counts]",
                                      50,
                                      -3.15,
                                      3.15,
                                      0.,
                                      1024.);
  meHitQvsPhi_[1] = ibook.bookProfile("BtlHitQvsPhiPlus",
                                      "BTL DIGI charge vs #phi (+ side);#phi_{DIGI} [rad];Q_{DIGI} [ADC counts]",
                                      50,
                                      -3.15,
                                      3.15,
                                      0.,
                                      1024.);
  meHitQvsEta_[0] = ibook.bookProfile("BtlHitQvsEtaMinus",
                                      "BTL DIGI charge vs #eta (- side);#eta_{DIGI};Q_{DIGI} [ADC counts]",
                                      50,
                                      -1.55,
                                      1.55,
                                      0.,
                                      1024.);
  meHitQvsEta_[1] = ibook.bookProfile("BtlHitQvsEtaPlus",
                                      "BTL DIGI charge vs #eta (+ side);#eta_{DIGI};Q_{DIGI} [ADC counts]",
                                      50,
                                      -1.55,
                                      1.55,
                                      0.,
                                      1024.);
  meHitQvsZ_[0] = ibook.bookProfile("BtlHitQvsZMinus",
                                    "BTL DIGI charge vs Z (- side);Z_{DIGI} [cm];Q_{DIGI} [ADC counts]",
                                    50,
                                    -260.,
                                    260.,
                                    0.,
                                    1024.);
  meHitQvsZ_[1] = ibook.bookProfile(
      "BtlHitQvsZPlus", "BTL DIGI charge vs Z (+ side);Z_{DIGI} [cm];Q_{DIGI} [ADC counts]", 50, -260., 260., 0., 1024.);

  meHitT1coarseVsPhi_[0] =
      ibook.bookProfile("BtlHitT1coarseVsPhiMinus",
                        "BTL DIGI T1 Coarse vs #phi (- side);#phi_{DIGI} [rad];T1Coarse_{DIGI} [# clk cycles]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        1024.);
  meHitT1coarseVsPhi_[1] =
      ibook.bookProfile("BtlHitT1coarseVsPhiPlus",
                        "BTL DIGI T1 Coarse vs #phi (+ side);#phi_{DIGI} [rad];T1Coarse_{DIGI} [# clk cycles]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        1024.);
  meHitT2coarseVsPhi_[0] =
      ibook.bookProfile("BtlHitT2coarseVsPhiMinus",
                        "BTL DIGI T2 Coarse vs #phi (- side);#phi_{DIGI} [rad];T2Coarse_{DIGI} [# clk cycles]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        1024.);
  meHitT2coarseVsPhi_[1] =
      ibook.bookProfile("BtlHitT2coarseVsPhiPlus",
                        "BTL DIGI T2 Coarse vs #phi (+ side);#phi_{DIGI} [rad];T2Coarse_{DIGI} [# clk cycles]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        1024.);
  meHitT1fineVsPhi_[0] =
      ibook.bookProfile("BtlHitT1fineVsPhiMinus",
                        "BTL DIGI T1 Fine vs #phi (- side);#phi_{DIGI} [rad];T1Fine_{DIGI} [TDC counts]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        1024.);
  meHitT1fineVsPhi_[1] =
      ibook.bookProfile("BtlHitT1fineVsPhiPlus",
                        "BTL DIGI T1 Fine vs #phi (+ side);#phi_{DIGI} [rad];T1Fine_{DIGI} [TDC counts]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        1024.);
  meHitT2fineVsPhi_[0] =
      ibook.bookProfile("BtlHitT2fineVsPhiMinus",
                        "BTL DIGI T2 Fine vs #phi (- side);#phi_{DIGI} [rad];T2Fine_{DIGI} [TDC counts]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        1024.);
  meHitT2fineVsPhi_[1] =
      ibook.bookProfile("BtlHitT2fineVsPhiPlus",
                        "BTL DIGI T2 Fine vs #phi (+ side);#phi_{DIGI} [rad];T2Fine_{DIGI} [TDC counts]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        1024.);

  meHitT1coarseVsEta_[0] =
      ibook.bookProfile("BtlHitT1coarseVsEtaMinus",
                        "BTL DIGI T1 Coarse vs #eta (- side);#eta_{DIGI};T1Coarse_{DIGI} [# clk cycles]",
                        50,
                        -1.55,
                        1.55,
                        0.,
                        1024.);
  meHitT1coarseVsEta_[1] =
      ibook.bookProfile("BtlHitT1coarseVsEtaPlus",
                        "BTL DIGI T1 Coarse vs #eta (+ side);#eta_{DIGI};T1Coarse_{DIGI} [# clk cycles]",
                        50,
                        -1.55,
                        1.55,
                        0.,
                        1024.);
  meHitT2coarseVsEta_[0] =
      ibook.bookProfile("BtlHitT2coarseVsEtaMinus",
                        "BTL DIGI T2 Coarse vs #eta (- side);#eta_{DIGI};T2Coarse_{DIGI} [# clk cycles]",
                        50,
                        -1.55,
                        1.55,
                        0.,
                        1024.);
  meHitT2coarseVsEta_[1] =
      ibook.bookProfile("BtlHitT2coarseVsEtaPlus",
                        "BTL DIGI T2 Coarse vs #eta (+ side);#eta_{DIGI};T2Coarse_{DIGI} [# clk cycles]",
                        50,
                        -1.55,
                        1.55,
                        0.,
                        1024.);
  meHitT1fineVsEta_[0] = ibook.bookProfile("BtlHitT1fineVsEtaMinus",
                                           "BTL DIGI T1 Fine vs #eta (- side);#eta_{DIGI};T1Fine_{DIGI} [TDC counts]",
                                           50,
                                           -1.55,
                                           1.55,
                                           0.,
                                           1024.);
  meHitT1fineVsEta_[1] = ibook.bookProfile("BtlHitT1fineVsEtaPlus",
                                           "BTL DIGI T1 Fine vs #eta (+ side);#eta_{DIGI};T1Fine_{DIGI} [TDC counts]",
                                           50,
                                           -1.55,
                                           1.55,
                                           0.,
                                           1024.);
  meHitT2fineVsEta_[0] = ibook.bookProfile("BtlHitT2fineVsEtaMinus",
                                           "BTL DIGI T2 Fine vs #eta (- side);#eta_{DIGI};T2Fine_{DIGI} [TDC counts]",
                                           50,
                                           -1.55,
                                           1.55,
                                           0.,
                                           1024.);
  meHitT2fineVsEta_[1] = ibook.bookProfile("BtlHitT2fineVsEtaPlus",
                                           "BTL DIGI T2 Fine vs #eta (+ side);#eta_{DIGI};T2Fine_{DIGI} [TDC counts]",
                                           50,
                                           -1.55,
                                           1.55,
                                           0.,
                                           1024.);

  meHitT1coarseVsZ_[0] =
      ibook.bookProfile("BtlHitT1coarseVsZMinus",
                        "BTL DIGI T1 Coarse vs Z (- side);Z_{DIGI} [cm];T1Coarse_{DIGI} [# clk cycles]",
                        50,
                        -260.,
                        260.,
                        0.,
                        1024.);
  meHitT1coarseVsZ_[1] =
      ibook.bookProfile("BtlHitT1coarseVsZPlus",
                        "BTL DIGI T1 Coarse vs Z (+ side);Z_{DIGI} [cm];T1Coarse_{DIGI} [# clk cycles]",
                        50,
                        -260.,
                        260.,
                        0.,
                        1024.);
  meHitT2coarseVsZ_[0] =
      ibook.bookProfile("BtlHitT2coarseVsZMinus",
                        "BTL DIGI T2 Coarse vs Z (- side);Z_{DIGI} [cm];T2Coarse_{DIGI} [# clk cycles]",
                        50,
                        -260.,
                        260.,
                        0.,
                        1024.);
  meHitT2coarseVsZ_[1] =
      ibook.bookProfile("BtlHitT2coarseVsZPlus",
                        "BTL DIGI T2 Coarse vs Z (+ side);Z_{DIGI} [cm];T2Coarse_{DIGI} [# clk cycles]",
                        50,
                        -260.,
                        260.,
                        0.,
                        1024.);
  meHitT1fineVsZ_[0] = ibook.bookProfile("BtlHitT1fineVsZMinus",
                                         "BTL DIGI T1 Fine vs Z (- side);Z_{DIGI} [cm];T1Fine_{DIGI} [TDC counts]",
                                         50,
                                         -260.,
                                         260.,
                                         0.,
                                         1024.);
  meHitT1fineVsZ_[1] = ibook.bookProfile("BtlHitT1fineVsZPlus",
                                         "BTL DIGI T1 Fine vs Z (+ side);Z_{DIGI} [cm];T1Fine_{DIGI} [TDC counts]",
                                         50,
                                         -260.,
                                         260.,
                                         0.,
                                         1024.);
  meHitT2fineVsZ_[0] = ibook.bookProfile("BtlHitT2fineVsZMinus",
                                         "BTL DIGI T2 Fine vs Z (- side);Z_{DIGI} [cm];T2Fine_{DIGI} [TDC counts]",
                                         50,
                                         -260.,
                                         260.,
                                         0.,
                                         1024.);
  meHitT2fineVsZ_[1] = ibook.bookProfile("BtlHitT2fineVsZPlus",
                                         "BTL DIGI T2 Fine vs Z (+ side);Z_{DIGI} [cm];T2Fine_{DIGI} [TDC counts]",
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
  desc.add<edm::InputTag>("inputTag", edm::InputTag("btlDigiSoAProducer", "MTDBarrelSoA"));
  desc.add<bool>("optionalPlots", false);

  descriptions.add("btlDigiSoAHitsDefaultValid", desc);
}

DEFINE_FWK_MODULE(BtlDigiSoAHitsValidation);
