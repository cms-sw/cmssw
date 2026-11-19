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
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/ValidHandle.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"

#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

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
  const bool optionalPlots_;

  edm::EDGetTokenT<BTLDigiCollection> btlDigiHitsToken_;

  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;

  // --- histograms declaration

  MonitorElement* meNhits_[2];

  MonitorElement* meHitCharge_[2];
  MonitorElement* meHitTime_[2];

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
    : folder_(iConfig.getParameter<std::string>("folder")),
      optionalPlots_(iConfig.getParameter<bool>("optionalPlots")) {
  btlDigiHitsToken_ = consumes<BTLDigiCollection>(iConfig.getParameter<edm::InputTag>("inputTag"));
  mtdgeoToken_ = esConsumes<MTDGeometry, MTDDigiGeometryRecord>();
  mtdtopoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>();
}

BtlDigiHitsValidation::~BtlDigiHitsValidation() {}

// ------------ method called for each event  ------------
void BtlDigiHitsValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  auto geometryHandle = iSetup.getTransientHandle(mtdgeoToken_);
  const MTDGeometry* geom = geometryHandle.product();

  auto topologyHandle = iSetup.getTransientHandle(mtdtopoToken_);
  const MTDTopology* topology = topologyHandle.product();

  auto btlDigiHitsHandle = makeValid(iEvent.getHandle(btlDigiHitsToken_));

  // --- Loop over the BLT DIGI hits

  unsigned int n_digi_btl[2] = {0, 0};

  for (const auto& dataFrame : *btlDigiHitsHandle) {
    BTLDetId detId = dataFrame.id();
    DetId geoId = detId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode()));
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

  if (n_digi_btl[0] > 0)
    meNhits_[0]->Fill(log10(n_digi_btl[0]));
  if (n_digi_btl[1] > 0)
    meNhits_[1]->Fill(log10(n_digi_btl[1]));
}

// ------------ method for histogram booking ------------
void BtlDigiHitsValidation::bookHistograms(DQMStore::IBooker& ibook,
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
  meHitTime_[0] = ibook.book1D("BtlHitTimeMinus", "BTL DIGI hits ToA (- side);ToA_{DIGI} [TDC counts]", 100, 0., 1024.);
  meHitTime_[1] = ibook.book1D("BtlHitTimePlus", "BTL DIGI hits ToA (+ side);ToA_{DIGI} [TDC counts]", 100, 0., 1024.);
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

  meHitTvsQ_[0] = ibook.bookProfile("BtlHitTvsQMinus",
                                    "BTL DIGI ToA vs charge (- side);Q_{DIGI} [ADC counts];ToA_{DIGI} [TDC counts]",
                                    50,
                                    0.,
                                    1024.,
                                    0.,
                                    1024.);
  meHitTvsQ_[1] = ibook.bookProfile("BtlHitTvsQPlus",
                                    "BTL DIGI ToA vs charge (+ side);Q_{DIGI} [ADC counts];ToA_{DIGI} [TDC counts]",
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
  meHitTvsPhi_[0] = ibook.bookProfile("BtlHitTvsPhiMinus",
                                      "BTL DIGI ToA vs #phi (- side);#phi_{DIGI} [rad];ToA_{DIGI} [TDC counts]",
                                      50,
                                      -3.15,
                                      3.15,
                                      0.,
                                      1024.);
  meHitTvsPhi_[1] = ibook.bookProfile("BtlHitTvsPhiPlus",
                                      "BTL DIGI ToA vs #phi (+ side);#phi_{DIGI} [rad];ToA_{DIGI} [TDC counts]",
                                      50,
                                      -3.15,
                                      3.15,
                                      0.,
                                      1024.);
  meHitTvsEta_[0] = ibook.bookProfile("BtlHitTvsEtaMinus",
                                      "BTL DIGI ToA vs #eta (- side);#eta_{DIGI};ToA_{DIGI} [TDC counts]",
                                      50,
                                      -1.55,
                                      1.55,
                                      0.,
                                      1024.);
  meHitTvsEta_[1] = ibook.bookProfile("BtlHitTvsEtaPlus",
                                      "BTL DIGI ToA vs #eta (+ side);#eta_{DIGI};ToA_{DIGI} [TDC counts]",
                                      50,
                                      -1.55,
                                      1.55,
                                      0.,
                                      1024.);
  meHitTvsZ_[0] = ibook.bookProfile(
      "BtlHitTvsZMinus", "BTL DIGI ToA vs Z (- side);Z_{DIGI} [cm];ToA_{DIGI} [TDC counts]", 50, -260., 260., 0., 1024.);
  meHitTvsZ_[1] = ibook.bookProfile(
      "BtlHitTvsZPlus", "BTL DIGI ToA vs Z (+ side);Z_{DIGI} [cm];ToA_{DIGI} [TDC counts]", 50, -260., 260., 0., 1024.);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BtlDigiHitsValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/BTL/DigiHits");
  desc.add<edm::InputTag>("inputTag", edm::InputTag("mix", "FTLBarrel"));
  desc.add<bool>("optionalPlots", false);

  descriptions.add("btlDigiHitsDefaultValid", desc);
}

DEFINE_FWK_MODULE(BtlDigiHitsValidation);
