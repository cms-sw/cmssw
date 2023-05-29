// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      EtlDigiHitsValidation
//
/**\class EtlDigiHitsValidation EtlDigiHitsValidation.cc Validation/MtdValidation/plugins/EtlDigiHitsValidation.cc

 Description: ETL DIGI hits validation

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
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

class EtlDigiHitsValidation : public DQMEDAnalyzer {
public:
  explicit EtlDigiHitsValidation(const edm::ParameterSet&);
  ~EtlDigiHitsValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ------------ member data ------------

  const std::string folder_;
  const bool optionalPlots_;

  edm::EDGetTokenT<ETLDigiCollection> etlDigiHitsToken_;

  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;

  // --- histograms declaration

  MonitorElement* meNhits_[4];
  MonitorElement* meNhitsPerLGAD_[4];

  MonitorElement* meHitCharge_[4];
  MonitorElement* meHitTime_[4];

  MonitorElement* meOccupancy_[4];

  MonitorElement* meLocalOccupancy_[2];  //folding the two ETL discs
  MonitorElement* meHitXlocal_[2];
  MonitorElement* meHitYlocal_[2];

  MonitorElement* meHitX_[4];
  MonitorElement* meHitY_[4];
  MonitorElement* meHitZ_[4];
  MonitorElement* meHitPhi_[4];
  MonitorElement* meHitEta_[4];

  MonitorElement* meHitTvsQ_[4];
  MonitorElement* meHitQvsPhi_[4];
  MonitorElement* meHitQvsEta_[4];
  MonitorElement* meHitTvsPhi_[4];
  MonitorElement* meHitTvsEta_[4];

  std::array<std::unordered_map<uint32_t, uint32_t>, 4> ndigiPerLGAD_;
};

// ------------ constructor and destructor --------------
EtlDigiHitsValidation::EtlDigiHitsValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      optionalPlots_(iConfig.getParameter<bool>("optionalPlots")) {
  etlDigiHitsToken_ = consumes<ETLDigiCollection>(iConfig.getParameter<edm::InputTag>("inputTag"));
  mtdgeoToken_ = esConsumes<MTDGeometry, MTDDigiGeometryRecord>();
  mtdtopoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>();
}

EtlDigiHitsValidation::~EtlDigiHitsValidation() {}

// ------------ method called for each event  ------------
void EtlDigiHitsValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  auto geometryHandle = iSetup.getTransientHandle(mtdgeoToken_);
  const MTDGeometry* geom = geometryHandle.product();

  auto topologyHandle = iSetup.getTransientHandle(mtdtopoToken_);
  const MTDTopology* topology = topologyHandle.product();

  bool topo1Dis = false;
  bool topo2Dis = false;
  if (MTDTopologyMode::etlLayoutFromTopoMode(topology->getMTDTopologyMode()) == ETLDetId::EtlLayout::tp) {
    topo1Dis = true;
  } else {
    topo2Dis = true;
  }

  auto etlDigiHitsHandle = makeValid(iEvent.getHandle(etlDigiHitsToken_));

  // --- Loop over the ETL DIGI hits

  unsigned int n_digi_etl[4] = {0, 0, 0, 0};
  for (size_t i = 0; i < 4; i++) {
    ndigiPerLGAD_[i].clear();
  }

  for (const auto& dataFrame : *etlDigiHitsHandle) {
    // --- Get the on-time sample
    int isample = 2;
    double weight = 1.0;
    const auto& sample = dataFrame.sample(isample);
    ETLDetId detId = dataFrame.id();
    DetId geoId = detId.geographicalId();

    const MTDGeomDet* thedet = geom->idToDet(geoId);
    if (thedet == nullptr)
      throw cms::Exception("EtlDigiHitsValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                    << detId.rawId() << ") is invalid!" << std::dec << std::endl;
    const PixelTopology& topo = static_cast<const PixelTopology&>(thedet->topology());

    Local3DPoint local_point(topo.localX(sample.row()), topo.localY(sample.column()), 0.);
    const auto& global_point = thedet->toGlobal(local_point);

    // --- Fill the histograms

    int idet = 999;

    if (topo1Dis) {
      if (detId.zside() == -1) {
        idet = 0;
      } else if (detId.zside() == 1) {
        idet = 2;
      } else {
        continue;
      }
    }

    if (topo2Dis) {
      if (detId.discSide() == 1) {
        weight = -weight;
      }
      if ((detId.zside() == -1) && (detId.nDisc() == 1)) {
        idet = 0;
      } else if ((detId.zside() == -1) && (detId.nDisc() == 2)) {
        idet = 1;
      } else if ((detId.zside() == 1) && (detId.nDisc() == 1)) {
        idet = 2;
      } else if ((detId.zside() == 1) && (detId.nDisc() == 2)) {
        idet = 3;
      } else {
        continue;
      }
    }

    meHitCharge_[idet]->Fill(sample.data());
    meHitTime_[idet]->Fill(sample.toa());
    meOccupancy_[idet]->Fill(global_point.x(), global_point.y(), weight);

    if (optionalPlots_) {
      if ((idet == 0) || (idet == 1)) {
        meLocalOccupancy_[0]->Fill(local_point.x(), local_point.y());
        meHitXlocal_[0]->Fill(local_point.x());
        meHitYlocal_[0]->Fill(local_point.y());

      } else if ((idet == 2) || (idet == 3)) {
        meLocalOccupancy_[1]->Fill(local_point.x(), local_point.y());
        meHitXlocal_[1]->Fill(local_point.x());
        meHitYlocal_[1]->Fill(local_point.y());
      }
    }

    meHitX_[idet]->Fill(global_point.x());
    meHitY_[idet]->Fill(global_point.y());
    meHitZ_[idet]->Fill(global_point.z());
    meHitPhi_[idet]->Fill(global_point.phi());
    meHitEta_[idet]->Fill(global_point.eta());

    meHitTvsQ_[idet]->Fill(sample.data(), sample.toa());
    meHitQvsPhi_[idet]->Fill(global_point.phi(), sample.data());
    meHitQvsEta_[idet]->Fill(global_point.eta(), sample.data());
    meHitTvsPhi_[idet]->Fill(global_point.phi(), sample.toa());
    meHitTvsEta_[idet]->Fill(global_point.eta(), sample.toa());

    n_digi_etl[idet]++;
    size_t ncount(0);
    ndigiPerLGAD_[idet].emplace(detId.rawId(), ncount);
    ndigiPerLGAD_[idet].at(detId.rawId())++;

  }  // dataFrame loop

  if (topo1Dis) {
    meNhits_[0]->Fill(log10(n_digi_etl[0]));
    meNhits_[2]->Fill(log10(n_digi_etl[2]));
    for (const auto& thisNdigi : ndigiPerLGAD_[0]) {
      meNhitsPerLGAD_[0]->Fill(thisNdigi.second);
    }
    for (const auto& thisNdigi : ndigiPerLGAD_[2]) {
      meNhitsPerLGAD_[2]->Fill(thisNdigi.second);
    }
  }

  if (topo2Dis) {
    for (int i = 0; i < 4; i++) {
      meNhits_[i]->Fill(log10(n_digi_etl[i]));
      for (const auto& thisNdigi : ndigiPerLGAD_[i]) {
        meNhitsPerLGAD_[i]->Fill(thisNdigi.second);
      }
    }
  }
}

// ------------ method for histogram booking ------------
void EtlDigiHitsValidation::bookHistograms(DQMStore::IBooker& ibook,
                                           edm::Run const& run,
                                           edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  // --- histograms booking

  meNhits_[0] = ibook.book1D("EtlNhitsZnegD1",
                             "Number of ETL DIGI hits (-Z, Single(topo1D)/First(topo2D) disk);log_{10}(N_{DIGI})",
                             100,
                             0.,
                             5.25);
  meNhits_[1] =
      ibook.book1D("EtlNhitsZnegD2", "Number of ETL DIGI hits (-Z, Second disk);log_{10}(N_{DIGI})", 100, 0., 5.25);
  meNhits_[2] = ibook.book1D("EtlNhitsZposD1",
                             "Number of ETL DIGI hits (+Z, Single(topo1D)/First(topo2D) disk);log_{10}(N_{DIGI})",
                             100,
                             0.,
                             5.25);
  meNhits_[3] =
      ibook.book1D("EtlNhitsZposD2", "Number of ETL DIGI hits (+Z, Second disk);log_{10}(N_{DIGI})", 100, 0., 5.25);

  meNhitsPerLGAD_[0] = ibook.book1D("EtlNhitsPerLGADZnegD1",
                                    "Number of ETL DIGI hits (-Z, Single(topo1D)/First(topo2D) disk) per LGAD;N_{DIGI}",
                                    50,
                                    0.,
                                    50.);
  meNhitsPerLGAD_[1] =
      ibook.book1D("EtlNhitsPerLGADZnegD2", "Number of ETL DIGI hits (-Z, Second disk) per LGAD;N_{DIGI}", 50, 0., 50.);
  meNhitsPerLGAD_[2] = ibook.book1D("EtlNhitsPerLGADZposD1",
                                    "Number of ETL DIGI hits (+Z, Single(topo1D)/First(topo2D) disk) per LGAD;N_{DIGI}",
                                    50,
                                    0.,
                                    50.);
  meNhitsPerLGAD_[3] =
      ibook.book1D("EtlNhitsPerLGADZposD2", "Number of ETL DIGI hits (+Z, Second disk) per LGAD;N_{DIGI}", 50, 0., 50.);

  meHitCharge_[0] = ibook.book1D("EtlHitChargeZnegD1",
                                 "ETL DIGI hits charge (-Z, Single(topo1D)/First(topo2D) disk);Q_{DIGI} [ADC counts]",
                                 100,
                                 0.,
                                 256.);
  meHitCharge_[1] =
      ibook.book1D("EtlHitChargeZnegD2", "ETL DIGI hits charge (-Z, Second disk);Q_{DIGI} [ADC counts]", 100, 0., 256.);
  meHitCharge_[2] = ibook.book1D("EtlHitChargeZposD1",
                                 "ETL DIGI hits charge (+Z, Single(topo1D)/First(topo2D) disk);Q_{DIGI} [ADC counts]",
                                 100,
                                 0.,
                                 256.);
  meHitCharge_[3] =
      ibook.book1D("EtlHitChargeZposD2", "ETL DIGI hits charge (+Z, Second disk);Q_{DIGI} [ADC counts]", 100, 0., 256.);
  meHitTime_[0] = ibook.book1D("EtlHitTimeZnegD1",
                               "ETL DIGI hits ToA (-Z, Single(topo1D)/First(topo2D) disk);ToA_{DIGI} [TDC counts]",
                               100,
                               0.,
                               2000.);
  meHitTime_[1] =
      ibook.book1D("EtlHitTimeZnegD2", "ETL DIGI hits ToA (-Z, Second disk);ToA_{DIGI} [TDC counts]", 100, 0., 2000.);
  meHitTime_[2] = ibook.book1D("EtlHitTimeZposD1",
                               "ETL DIGI hits ToA (+Z, Single(topo1D)/First(topo2D) disk);ToA_{DIGI} [TDC counts]",
                               100,
                               0.,
                               2000.);
  meHitTime_[3] =
      ibook.book1D("EtlHitTimeZposD2", "ETL DIGI hits ToA (+Z, Second disk);ToA_{DIGI} [TDC counts]", 100, 0., 2000.);

  meOccupancy_[0] =
      ibook.book2D("EtlOccupancyZnegD1",
                   "ETL DIGI hits occupancy (-Z, Single(topo1D)/First(topo2D) disk);X_{DIGI} [cm];Y_{DIGI} [cm]",
                   135,
                   -135.,
                   135.,
                   135,
                   -135.,
                   135.);
  meOccupancy_[1] = ibook.book2D("EtlOccupancyZnegD2",
                                 "ETL DIGI hits occupancy (-Z, Second disk);X_{DIGI} [cm];Y_{DIGI} [cm]",
                                 135,
                                 -135.,
                                 135.,
                                 135,
                                 -135.,
                                 135.);
  meOccupancy_[2] =
      ibook.book2D("EtlOccupancyZposD1",
                   "ETL DIGI hits occupancy (+Z, Single(topo1D)/First(topo2D) disk);X_{DIGI} [cm];Y_{DIGI} [cm]",
                   135,
                   -135.,
                   135.,
                   135,
                   -135.,
                   135.);
  meOccupancy_[3] = ibook.book2D("EtlOccupancyZposD2",
                                 "ETL DIGI hits occupancy (+Z, Second disk);X_{DIGI} [cm];Y_{DIGI} [cm]",
                                 135,
                                 -135.,
                                 135.,
                                 135,
                                 -135.,
                                 135.);
  if (optionalPlots_) {
    meLocalOccupancy_[0] = ibook.book2D("EtlLocalOccupancyZneg",
                                        "ETL DIGI hits local occupancy (-Z);X_{DIGI} [cm];Y_{DIGI} [cm]",
                                        100,
                                        -2.2,
                                        2.2,
                                        50,
                                        -1.1,
                                        1.1);
    meLocalOccupancy_[1] = ibook.book2D("EtlLocalOccupancyZpos",
                                        "ETL DIGI hits local occupancy (+Z);X_{DIGI} [cm];Y_{DIGI} [cm]",
                                        100,
                                        -2.2,
                                        2.2,
                                        50,
                                        -1.1,
                                        1.1);
    meHitXlocal_[0] = ibook.book1D("EtlHitXlocalZneg", "ETL DIGI local X (-Z);X_{DIGI}^{LOC} [cm]", 100, -2.2, 2.2);
    meHitXlocal_[1] = ibook.book1D("EtlHitXlocalZpos", "ETL DIGI local X (+Z);X_{DIGI}^{LOC} [cm]", 100, -2.2, 2.2);
    meHitYlocal_[0] = ibook.book1D("EtlHitYlocalZneg", "ETL DIGI local Y (-Z);Y_{DIGI}^{LOC} [cm]", 50, -1.1, 1.1);
    meHitYlocal_[1] = ibook.book1D("EtlHitYlocalZpos", "ETL DIGI local Y (-Z);Y_{DIGI}^{LOC} [cm]", 50, -1.1, 1.1);
  }
  meHitX_[0] = ibook.book1D(
      "EtlHitXZnegD1", "ETL DIGI hits X (-Z, Single(topo1D)/First(topo2D) disk);X_{DIGI} [cm]", 100, -130., 130.);
  meHitX_[1] = ibook.book1D("EtlHitXZnegD2", "ETL DIGI hits X (-Z, Second disk);X_{DIGI} [cm]", 100, -130., 130.);
  meHitX_[2] = ibook.book1D(
      "EtlHitXZposD1", "ETL DIGI hits X (+Z, Single(topo1D)/First(topo2D) disk);X_{DIGI} [cm]", 100, -130., 130.);
  meHitX_[3] = ibook.book1D("EtlHitXZposD2", "ETL DIGI hits X (+Z, Second disk);X_{DIGI} [cm]", 100, -130., 130.);
  meHitY_[0] = ibook.book1D(
      "EtlHitYZnegD1", "ETL DIGI hits Y (-Z, Single(topo1D)/First(topo2D) disk);Y_{DIGI} [cm]", 100, -130., 130.);
  meHitY_[1] = ibook.book1D("EtlHitYZnegD2", "ETL DIGI hits Y (-Z, Second disk);Y_{DIGI} [cm]", 100, -130., 130.);
  meHitY_[2] = ibook.book1D(
      "EtlHitYZposD1", "ETL DIGI hits Y (+Z, Single(topo1D)/First(topo2D) disk);Y_{DIGI} [cm]", 100, -130., 130.);
  meHitY_[3] = ibook.book1D("EtlHitYZposD2", "ETL DIGI hits Y (+Z, Second disk);Y_{DIGI} [cm]", 100, -130., 130.);
  meHitZ_[0] = ibook.book1D(
      "EtlHitZZnegD1", "ETL DIGI hits Z (-Z, Single(topo1D)/First(topo2D) disk);Z_{DIGI} [cm]", 100, -302., -298.);
  meHitZ_[1] = ibook.book1D("EtlHitZZnegD2", "ETL DIGI hits Z (-Z, Second disk);Z_{DIGI} [cm]", 100, -304., -300.);
  meHitZ_[2] = ibook.book1D(
      "EtlHitZZposD1", "ETL DIGI hits Z (+Z, Single(topo1D)/First(topo2D) disk);Z_{DIGI} [cm]", 100, 298., 302.);
  meHitZ_[3] = ibook.book1D("EtlHitZZposD2", "ETL DIGI hits Z (+Z, Second disk);Z_{DIGI} [cm]", 100, 300., 304.);

  meHitPhi_[0] = ibook.book1D("EtlHitPhiZnegD1",
                              "ETL DIGI hits #phi (-Z, Single(topo1D)/First(topo2D) disk);#phi_{DIGI} [rad]",
                              100,
                              -3.15,
                              3.15);
  meHitPhi_[1] =
      ibook.book1D("EtlHitPhiZnegD2", "ETL DIGI hits #phi (-Z, Second disk);#phi_{DIGI} [rad]", 100, -3.15, 3.15);
  meHitPhi_[2] = ibook.book1D("EtlHitPhiZposD1",
                              "ETL DIGI hits #phi (+Z, Single(topo1D)/First(topo2D) disk);#phi_{DIGI} [rad]",
                              100,
                              -3.15,
                              3.15);
  meHitPhi_[3] =
      ibook.book1D("EtlHitPhiZposD2", "ETL DIGI hits #phi (+Z, Second disk);#phi_{DIGI} [rad]", 100, -3.15, 3.15);
  meHitEta_[0] = ibook.book1D(
      "EtlHitEtaZnegD1", "ETL DIGI hits #eta (-Z, Single(topo1D)/First(topo2D) disk);#eta_{DIGI}", 100, -3.2, -1.56);
  meHitEta_[1] = ibook.book1D("EtlHitEtaZnegD2", "ETL DIGI hits #eta (-Z, Second disk);#eta_{DIGI}", 100, -3.2, -1.56);
  meHitEta_[2] = ibook.book1D(
      "EtlHitEtaZposD1", "ETL DIGI hits #eta (+Z, Single(topo1D)/First(topo2D) disk);#eta_{DIGI}", 100, 1.56, 3.2);
  meHitEta_[3] = ibook.book1D("EtlHitEtaZposD2", "ETL DIGI hits #eta (+Z, Second disk);#eta_{DIGI}", 100, 1.56, 3.2);
  meHitTvsQ_[0] = ibook.bookProfile(
      "EtlHitTvsQZnegD1",
      "ETL DIGI ToA vs charge (-Z, Single(topo1D)/First(topo2D) disk);Q_{DIGI} [ADC counts];ToA_{DIGI} [TDC counts]",
      50,
      0.,
      256.,
      0.,
      1024.);
  meHitTvsQ_[1] =
      ibook.bookProfile("EtlHitTvsQZnegD2",
                        "ETL DIGI ToA vs charge (-Z, Second Disk);Q_{DIGI} [ADC counts];ToA_{DIGI} [TDC counts]",
                        50,
                        0.,
                        256.,
                        0.,
                        1024.);
  meHitTvsQ_[2] = ibook.bookProfile(
      "EtlHitTvsQZposD1",
      "ETL DIGI ToA vs charge (+Z, Single(topo1D)/First(topo2D) disk);Q_{DIGI} [ADC counts];ToA_{DIGI} [TDC counts]",
      50,
      0.,
      256.,
      0.,
      1024.);
  meHitTvsQ_[3] =
      ibook.bookProfile("EtlHitTvsQZposD2",
                        "ETL DIGI ToA vs charge (+Z, Second disk);Q_{DIGI} [ADC counts];ToA_{DIGI} [TDC counts]",
                        50,
                        0.,
                        256.,
                        0.,
                        1024.);
  meHitQvsPhi_[0] = ibook.bookProfile(
      "EtlHitQvsPhiZnegD1",
      "ETL DIGI charge vs #phi (-Z, Single(topo1D)/First(topo2D) disk);#phi_{DIGI} [rad];Q_{DIGI} [ADC counts]",
      50,
      -3.15,
      3.15,
      0.,
      1024.);
  meHitQvsPhi_[1] =
      ibook.bookProfile("EtlHitQvsPhiZnegD2",
                        "ETL DIGI charge vs #phi (-Z, Second disk);#phi_{DIGI} [rad];Q_{DIGI} [ADC counts]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        1024.);
  meHitQvsPhi_[2] = ibook.bookProfile(
      "EtlHitQvsPhiZposD1",
      "ETL DIGI charge vs #phi (+Z, Single(topo1D)/First(topo2D) disk);#phi_{DIGI} [rad];Q_{DIGI} [ADC counts]",
      50,
      -3.15,
      3.15,
      0.,
      1024.);
  meHitQvsPhi_[3] =
      ibook.bookProfile("EtlHitQvsPhiZposD2",
                        "ETL DIGI charge vs #phi (+Z, Second disk);#phi_{DIGI} [rad];Q_{DIGI} [ADC counts]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        1024.);
  meHitQvsEta_[0] = ibook.bookProfile(
      "EtlHitQvsEtaZnegD1",
      "ETL DIGI charge vs #eta (-Z, Single(topo1D)/First(topo2D) disk);#eta_{DIGI};Q_{DIGI} [ADC counts]",
      50,
      -3.2,
      -1.56,
      0.,
      1024.);
  meHitQvsEta_[1] = ibook.bookProfile("EtlHitQvsEtaZnegD2",
                                      "ETL DIGI charge vs #eta (-Z, Second disk);#eta_{DIGI};Q_{DIGI} [ADC counts]",
                                      50,
                                      -3.2,
                                      -1.56,
                                      0.,
                                      1024.);
  meHitQvsEta_[2] = ibook.bookProfile(
      "EtlHitQvsEtaZposD1",
      "ETL DIGI charge vs #eta (+Z, Single(topo1D)/First(topo2D) disk);#eta_{DIGI};Q_{DIGI} [ADC counts]",
      50,
      1.56,
      3.2,
      0.,
      1024.);
  meHitQvsEta_[3] = ibook.bookProfile("EtlHitQvsEtaZposD2",
                                      "ETL DIGI charge vs #eta (+Z, Second disk);#eta_{DIGI};Q_{DIGI} [ADC counts]",
                                      50,
                                      1.56,
                                      3.2,
                                      0.,
                                      1024.);
  meHitTvsPhi_[0] = ibook.bookProfile(
      "EtlHitTvsPhiZnegD1",
      "ETL DIGI ToA vs #phi (-Z, Single(topo1D)/First(topo2D) disk);#phi_{DIGI} [rad];ToA_{DIGI} [TDC counts]",
      50,
      -3.15,
      3.15,
      0.,
      1024.);
  meHitTvsPhi_[1] =
      ibook.bookProfile("EtlHitTvsPhiZnegD2",
                        "ETL DIGI ToA vs #phi (-Z, Second disk);#phi_{DIGI} [rad];ToA_{DIGI} [TDC counts]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        1024.);
  meHitTvsPhi_[2] = ibook.bookProfile(
      "EtlHitTvsPhiZposD1",
      "ETL DIGI ToA vs #phi (+Z, Single(topo1D)/First(topo2D) disk);#phi_{DIGI} [rad];ToA_{DIGI} [TDC counts]",
      50,
      -3.15,
      3.15,
      0.,
      1024.);
  meHitTvsPhi_[3] =
      ibook.bookProfile("EtlHitTvsPhiZposD2",
                        "ETL DIGI ToA vs #phi (+Z, Second disk);#phi_{DIGI} [rad];ToA_{DIGI} [TDC counts]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        1024.);
  meHitTvsEta_[0] = ibook.bookProfile(
      "EtlHitTvsEtaZnegD1",
      "ETL DIGI ToA vs #eta (-Z, Single(topo1D)/First(topo2D) disk);#eta_{DIGI};ToA_{DIGI} [TDC counts]",
      50,
      -3.2,
      -1.56,
      0.,
      1024.);
  meHitTvsEta_[1] = ibook.bookProfile("EtlHitTvsEtaZnegD2",
                                      "ETL DIGI ToA vs #eta (-Z, Second disk);#eta_{DIGI};ToA_{DIGI} [TDC counts]",
                                      50,
                                      -3.2,
                                      -1.56,
                                      0.,
                                      1024.);
  meHitTvsEta_[2] = ibook.bookProfile(
      "EtlHitTvsEtaZposD1",
      "ETL DIGI ToA vs #eta (+Z, Single(topo1D)/First(topo2D) disk);#eta_{DIGI};ToA_{DIGI} [TDC counts]",
      50,
      1.56,
      3.2,
      0.,
      1024.);
  meHitTvsEta_[3] = ibook.bookProfile("EtlHitTvsEtaZposD2",
                                      "ETL DIGI ToA vs #eta (+Z, Second disk);#eta_{DIGI};ToA_{DIGI} [TDC counts]",
                                      50,
                                      1.56,
                                      3.2,
                                      0.,
                                      1024.);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void EtlDigiHitsValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/ETL/DigiHits");
  desc.add<edm::InputTag>("inputTag", edm::InputTag("mix", "FTLEndcap"));
  desc.add<bool>("optionalPlots", false);

  descriptions.add("etlDigiHitsDefaultValid", desc);
}

DEFINE_FWK_MODULE(EtlDigiHitsValidation);
