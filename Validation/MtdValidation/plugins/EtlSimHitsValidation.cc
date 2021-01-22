// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      EtlSimHitsValidation
//
/**\class EtlSimHitsValidation EtlSimHitsValidation.cc Validation/MtdValidation/plugins/EtlSimHitsValidation.cc

 Description: ETL SIM hits validation

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
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

struct MTDHit {
  float energy;
  float time;
  float x;
  float y;
  float z;
};

class EtlSimHitsValidation : public DQMEDAnalyzer {
public:
  explicit EtlSimHitsValidation(const edm::ParameterSet&);
  ~EtlSimHitsValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ------------ member data ------------

  const std::string folder_;
  const float hitMinEnergy1Dis_;
  const float hitMinEnergy2Dis_;

  edm::EDGetTokenT<CrossingFrame<PSimHit> > etlSimHitsToken_;

  // --- histograms declaration

  MonitorElement* meNhits_[4];
  MonitorElement* meNtrkPerCell_[4];

  MonitorElement* meHitEnergy_[4];
  MonitorElement* meHitTime_[4];

  MonitorElement* meHitXlocal_[4];
  MonitorElement* meHitYlocal_[4];
  MonitorElement* meHitZlocal_[4];

  MonitorElement* meOccupancy_[4];

  MonitorElement* meHitX_[4];
  MonitorElement* meHitY_[4];
  MonitorElement* meHitZ_[4];
  MonitorElement* meHitPhi_[4];
  MonitorElement* meHitEta_[4];

  MonitorElement* meHitTvsE_[4];
  MonitorElement* meHitEvsPhi_[4];
  MonitorElement* meHitEvsEta_[4];
  MonitorElement* meHitTvsPhi_[4];
  MonitorElement* meHitTvsEta_[4];
};

// ------------ constructor and destructor --------------
EtlSimHitsValidation::EtlSimHitsValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      hitMinEnergy1Dis_(iConfig.getParameter<double>("hitMinimumEnergy1Dis")),
      hitMinEnergy2Dis_(iConfig.getParameter<double>("hitMinimumEnergy2Dis")) {
  etlSimHitsToken_ = consumes<CrossingFrame<PSimHit> >(iConfig.getParameter<edm::InputTag>("inputTag"));
}

EtlSimHitsValidation::~EtlSimHitsValidation() {}

// ------------ method called for each event  ------------
void EtlSimHitsValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace geant_units::operators;

  edm::ESHandle<MTDGeometry> geometryHandle;
  iSetup.get<MTDDigiGeometryRecord>().get(geometryHandle);
  const MTDGeometry* geom = geometryHandle.product();

  edm::ESHandle<MTDTopology> topologyHandle;
  iSetup.get<MTDTopologyRcd>().get(topologyHandle);
  const MTDTopology* topology = topologyHandle.product();

  bool topo1Dis = false;
  bool topo2Dis = false;
  if (topology->getMTDTopologyMode() <= static_cast<int>(MTDTopologyMode::Mode::barphiflat)) {
    topo1Dis = true;
  }
  if (topology->getMTDTopologyMode() > static_cast<int>(MTDTopologyMode::Mode::barphiflat)) {
    topo2Dis = true;
  }

  auto etlSimHitsHandle = makeValid(iEvent.getHandle(etlSimHitsToken_));
  MixCollection<PSimHit> etlSimHits(etlSimHitsHandle.product());

  std::unordered_map<uint32_t, MTDHit> m_etlHits[4];
  std::unordered_map<uint32_t, std::set<int> > m_etlTrkPerCell[4];

  // --- Loop over the ETL SIM hits

  int idet = 999;

  for (auto const& simHit : etlSimHits) {
    // --- Use only hits compatible with the in-time bunch-crossing
    if (simHit.tof() < 0 || simHit.tof() > 25.)
      continue;

    ETLDetId id = simHit.detUnitId();
    if (topo1Dis) {
      if (id.zside() == -1) {
        idet = 0;
      } else if (id.zside() == 1) {
        idet = 2;
      } else {
        continue;
      }
    }

    if (topo2Dis) {
      if ((id.zside() == -1) && (id.nDisc() == 1)) {
        idet = 0;
      } else if ((id.zside() == -1) && (id.nDisc() == 2)) {
        idet = 1;
      } else if ((id.zside() == 1) && (id.nDisc() == 1)) {
        idet = 2;
      } else if ((id.zside() == 1) && (id.nDisc() == 2)) {
        idet = 3;
      } else {
        continue;
      }
    }

    m_etlTrkPerCell[idet][id.rawId()].insert(simHit.trackId());

    auto simHitIt = m_etlHits[idet].emplace(id.rawId(), MTDHit()).first;

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

  for (int idet = 0; idet < 4; ++idet) {  //two disks per side
    if (((idet == 1) || (idet == 3)) && (topo1Dis == true))
      continue;
    meNhits_[idet]->Fill(m_etlHits[idet].size());

    for (auto const& hit : m_etlTrkPerCell[idet]) {
      meNtrkPerCell_[idet]->Fill((hit.second).size());
    }

    for (auto const& hit : m_etlHits[idet]) {
      double weight = 1.0;
      if (topo1Dis) {
        if ((hit.second).energy < hitMinEnergy1Dis_)
          continue;
      }
      if (topo2Dis) {
        if ((hit.second).energy < hitMinEnergy2Dis_)
          continue;
      }
      // --- Get the SIM hit global position
      ETLDetId detId(hit.first);
      DetId geoId = detId.geographicalId();
      const MTDGeomDet* thedet = geom->idToDet(geoId);
      if (thedet == nullptr)
        throw cms::Exception("EtlSimHitsValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                     << detId.rawId() << ") is invalid!" << std::dec << std::endl;

      Local3DPoint local_point(
          convertMmToCm((hit.second).x), convertMmToCm((hit.second).y), convertMmToCm((hit.second).z));
      const auto& global_point = thedet->toGlobal(local_point);

      if (topo2Dis && (detId.discSide() == 1)) {
        weight = -weight;
      }

      // --- Fill the histograms

      meHitEnergy_[idet]->Fill((hit.second).energy);
      meHitTime_[idet]->Fill((hit.second).time);
      meHitXlocal_[idet]->Fill((hit.second).x);
      meHitYlocal_[idet]->Fill((hit.second).y);
      meHitZlocal_[idet]->Fill((hit.second).z);
      meOccupancy_[idet]->Fill(global_point.x(), global_point.y(), weight);
      meHitX_[idet]->Fill(global_point.x());
      meHitY_[idet]->Fill(global_point.y());
      meHitZ_[idet]->Fill(global_point.z());
      meHitPhi_[idet]->Fill(global_point.phi());
      meHitEta_[idet]->Fill(global_point.eta());
      meHitTvsE_[idet]->Fill((hit.second).energy, (hit.second).time);
      meHitEvsPhi_[idet]->Fill(global_point.phi(), (hit.second).energy);
      meHitEvsEta_[idet]->Fill(global_point.eta(), (hit.second).energy);
      meHitTvsPhi_[idet]->Fill(global_point.phi(), (hit.second).time);
      meHitTvsEta_[idet]->Fill(global_point.eta(), (hit.second).time);

    }  // hit loop

  }  // idet loop
}

// ------------ method for histogram booking ------------
void EtlSimHitsValidation::bookHistograms(DQMStore::IBooker& ibook,
                                          edm::Run const& run,
                                          edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  // --- histograms booking

  meNhits_[0] = ibook.book1D("EtlNhitsZnegD1",
                             "Number of ETL cells with SIM hits (-Z, Single(topo1D)/First(topo2D) disk);N_{ETL cells}",
                             100,
                             0.,
                             5000.);
  meNhits_[1] = ibook.book1D(
      "EtlNhitsZnegD2", "Number of ETL cells with SIM hits (-Z, Second disk);N_{ETL cells}", 100, 0., 5000.);
  meNhits_[2] = ibook.book1D("EtlNhitsZposD1",
                             "Number of ETL cells with SIM hits (+Z, Single(topo1D)/First(topo2D) disk);N_{ETL cells}",
                             100,
                             0.,
                             5000.);
  meNhits_[3] = ibook.book1D(
      "EtlNhitsZposD2", "Number of ETL cells with SIM hits (+Z, Second Disk);N_{ETL cells}", 100, 0., 5000.);
  meNtrkPerCell_[0] = ibook.book1D("EtlNtrkPerCellZnegD1",
                                   "Number of tracks per ETL sensor (-Z, Single(topo1D)/First(topo2D) disk);N_{trk}",
                                   10,
                                   0.,
                                   10.);
  meNtrkPerCell_[1] =
      ibook.book1D("EtlNtrkPerCellZnegD2", "Number of tracks per ETL sensor (-Z, Second disk);N_{trk}", 10, 0., 10.);
  meNtrkPerCell_[2] = ibook.book1D("EtlNtrkPerCellZposD1",
                                   "Number of tracks per ETL sensor (+Z, Single(topo1D)/First(topo2D) disk);N_{trk}",
                                   10,
                                   0.,
                                   10.);
  meNtrkPerCell_[3] =
      ibook.book1D("EtlNtrkPerCellZposD2", "Number of tracks per ETL sensor (+Z, Second disk);N_{trk}", 10, 0., 10.);
  meHitEnergy_[0] = ibook.book1D(
      "EtlHitEnergyZnegD1", "ETL SIM hits energy (-Z, Single(topo1D)/First(topo2D) disk);E_{SIM} [MeV]", 100, 0., 3.);
  meHitEnergy_[1] =
      ibook.book1D("EtlHitEnergyZnegD2", "ETL SIM hits energy (-Z, Second disk);E_{SIM} [MeV]", 100, 0., 3.);
  meHitEnergy_[2] = ibook.book1D(
      "EtlHitEnergyZposD1", "ETL SIM hits energy (+Z, Single(topo1D)/First(topo2D) disk);E_{SIM} [MeV]", 100, 0., 3.);
  meHitEnergy_[3] =
      ibook.book1D("EtlHitEnergyZposD2", "ETL SIM hits energy (+Z, Second disk);E_{SIM} [MeV]", 100, 0., 3.);
  meHitTime_[0] = ibook.book1D(
      "EtlHitTimeZnegD1", "ETL SIM hits ToA (-Z, Single(topo1D)/First(topo2D) disk);ToA_{SIM} [ns]", 100, 0., 25.);
  meHitTime_[1] = ibook.book1D("EtlHitTimeZnegD2", "ETL SIM hits ToA (-Z, Second disk);ToA_{SIM} [ns]", 100, 0., 25.);
  meHitTime_[2] = ibook.book1D(
      "EtlHitTimeZposD1", "ETL SIM hits ToA (+Z, Single(topo1D)/First(topo2D) disk);ToA_{SIM} [ns]", 100, 0., 25.);
  meHitTime_[3] = ibook.book1D("EtlHitTimeZposD2", "ETL SIM hits ToA (+Z, Second disk);ToA_{SIM} [ns]", 100, 0., 25.);

  meHitXlocal_[0] = ibook.book1D("EtlHitXlocalZnegD1",
                                 "ETL SIM local X (-Z, Single(topo1D)/First(topo2D) disk);X_{SIM}^{LOC} [mm]",
                                 100,
                                 -25.,
                                 25.);
  meHitXlocal_[1] =
      ibook.book1D("EtlHitXlocalZnegD2", "ETL SIM local X (-Z, Second disk);X_{SIM}^{LOC} [mm]", 100, -25., 25.);
  meHitXlocal_[2] = ibook.book1D("EtlHitXlocalZposD1",
                                 "ETL SIM local X (+Z, Single(topo1D)/First(topo2D) disk);X_{SIM}^{LOC} [mm]",
                                 100,
                                 -25.,
                                 25.);
  meHitXlocal_[3] =
      ibook.book1D("EtlHitXlocalZposD2", "ETL SIM local X (+Z, Second disk);X_{SIM}^{LOC} [mm]", 100, -25., 25.);

  meHitYlocal_[0] = ibook.book1D("EtlHitYlocalZnegD1",
                                 "ETL SIM local Y (-Z, Single(topo1D)/First(topo2D) disk);Y_{SIM}^{LOC} [mm]",
                                 100,
                                 -48.,
                                 48.);
  meHitYlocal_[1] =
      ibook.book1D("EtlHitYlocalZnegD2", "ETL SIM local Y (-Z, Second Disk);Y_{SIM}^{LOC} [mm]", 100, -48., 48.);
  meHitYlocal_[2] = ibook.book1D("EtlHitYlocalZposD1",
                                 "ETL SIM local Y (+Z, Single(topo1D)/First(topo2D) disk);Y_{SIM}^{LOC} [mm]",
                                 100,
                                 -48.,
                                 48.);
  meHitYlocal_[3] =
      ibook.book1D("EtlHitYlocalZposD2", "ETL SIM local Y (+Z, Second disk);Y_{SIM}^{LOC} [mm]", 100, -48., 48.);
  meHitZlocal_[0] = ibook.book1D("EtlHitZlocalZnegD1",
                                 "ETL SIM local Z (-Z, Single(topo1D)/First(topo2D) disk);Z_{SIM}^{LOC} [mm]",
                                 80,
                                 -0.16,
                                 0.16);
  meHitZlocal_[1] =
      ibook.book1D("EtlHitZlocalZnegD2", "ETL SIM local Z (-Z, Second disk);Z_{SIM}^{LOC} [mm]", 80, -0.16, 0.16);
  meHitZlocal_[2] = ibook.book1D("EtlHitZlocalZposD1",
                                 "ETL SIM local Z (+Z, Single(topo1D)/First(topo2D) disk);Z_{SIM}^{LOC} [mm]",
                                 80,
                                 -0.16,
                                 0.16);
  meHitZlocal_[3] =
      ibook.book1D("EtlHitZlocalZposD2", "ETL SIM local Z (+Z, Second disk);Z_{SIM}^{LOC} [mm]", 80, -0.16, 0.16);

  meOccupancy_[0] =
      ibook.book2D("EtlOccupancyZnegD1",
                   "ETL SIM hits occupancy (-Z, Single(topo1D)/First(topo2D) disk);X_{SIM} [cm];Y_{SIM} [cm]",
                   135,
                   -135.,
                   135.,
                   135,
                   -135.,
                   135.);
  meOccupancy_[1] = ibook.book2D("EtlOccupancyZnegD2",
                                 "ETL SIM hits occupancy (-Z, Second disk);X_{SIM} [cm];Y_{SIM} [cm]",
                                 135,
                                 -135.,
                                 135.,
                                 135,
                                 -135.,
                                 135.);
  meOccupancy_[2] =
      ibook.book2D("EtlOccupancyZposD1",
                   "ETL SIM hits occupancy (+Z, Single(topo1D)/First(topo2D) disk);X_{SIM} [cm];Y_{SIM} [cm]",
                   135,
                   -135.,
                   135.,
                   135,
                   -135.,
                   135.);
  meOccupancy_[3] = ibook.book2D("EtlOccupancyZposD2",
                                 "ETL SIM hits occupancy (+Z, Second disk);X_{SIM} [cm];Y_{SIM} [cm]",
                                 135,
                                 -135.,
                                 135.,
                                 135,
                                 -135.,
                                 135.);

  meHitX_[0] = ibook.book1D(
      "EtlHitXZnegD1", "ETL SIM hits X (+Z, Single(topo1D)/First(topo2D) disk);X_{SIM} [cm]", 100, -130., 130.);
  meHitX_[1] = ibook.book1D("EtlHitXZnegD2", "ETL SIM hits X (-Z, Second disk);X_{SIM} [cm]", 100, -130., 130.);
  meHitX_[2] = ibook.book1D(
      "EtlHitXZposD1", "ETL SIM hits X (+Z, Single(topo1D)/First(topo2D) disk);X_{SIM} [cm]", 100, -130., 130.);
  meHitX_[3] = ibook.book1D("EtlHitXZposD2", "ETL SIM hits X (+Z, Second disk);X_{SIM} [cm]", 100, -130., 130.);
  meHitY_[0] = ibook.book1D(
      "EtlHitYZnegD1", "ETL SIM hits Y (-Z, Single(topo1D)/First(topo2D) disk);Y_{SIM} [cm]", 100, -130., 130.);
  meHitY_[1] = ibook.book1D("EtlHitYZnegD2", "ETL SIM hits Y (-Z, Second disk);Y_{SIM} [cm]", 100, -130., 130.);
  meHitY_[2] = ibook.book1D(
      "EtlHitYZposD1", "ETL SIM hits Y (+Z, Single(topo1D)/First(topo2D) disk);Y_{SIM} [cm]", 100, -130., 130.);
  meHitY_[3] = ibook.book1D("EtlHitYZposD2", "ETL SIM hits Y (+Z, Second disk);Y_{SIM} [cm]", 100, -130., 130.);
  meHitZ_[0] = ibook.book1D(
      "EtlHitZZnegD1", "ETL SIM hits Z (-Z, Single(topo1D)/First(topo2D) disk);Z_{SIM} [cm]", 100, -302., -298.);
  meHitZ_[1] = ibook.book1D("EtlHitZZnegD2", "ETL SIM hits Z (-Z, Second disk);Z_{SIM} [cm]", 100, -304., -300.);
  meHitZ_[2] = ibook.book1D(
      "EtlHitZZposD1", "ETL SIM hits Z (+Z, Single(topo1D)/First(topo2D) disk);Z_{SIM} [cm]", 100, 298., 302.);
  meHitZ_[3] = ibook.book1D("EtlHitZZposD2", "ETL SIM hits Z (+Z, Second disk);Z_{SIM} [cm]", 100, 300., 304.);

  meHitPhi_[0] = ibook.book1D(
      "EtlHitPhiZnegD1", "ETL SIM hits #phi (-Z, Single(topo1D)/First(topo2D) disk);#phi_{SIM} [rad]", 100, -3.15, 3.15);
  meHitPhi_[1] =
      ibook.book1D("EtlHitPhiZnegD2", "ETL SIM hits #phi (-Z, Second disk);#phi_{SIM} [rad]", 100, -3.15, 3.15);
  meHitPhi_[2] = ibook.book1D(
      "EtlHitPhiZposD1", "ETL SIM hits #phi (+Z, Single(topo1D)/First(topo2D) disk);#phi_{SIM} [rad]", 100, -3.15, 3.15);
  meHitPhi_[3] =
      ibook.book1D("EtlHitPhiZposD2", "ETL SIM hits #phi (+Z, Second disk);#phi_{SIM} [rad]", 100, -3.15, 3.15);
  meHitEta_[0] = ibook.book1D(
      "EtlHitEtaZnegD1", "ETL SIM hits #eta (-Z, Single(topo1D)/First(topo2D) disk);#eta_{SIM}", 100, -3.2, -1.56);
  meHitEta_[1] = ibook.book1D("EtlHitEtaZnegD2", "ETL SIM hits #eta (-Z, Second disk);#eta_{SIM}", 100, -3.2, -1.56);
  meHitEta_[2] = ibook.book1D(
      "EtlHitEtaZposD1", "ETL SIM hits #eta (+Z, Single(topo1D)/First(topo2D) disk);#eta_{SIM}", 100, 1.56, 3.2);
  meHitEta_[3] = ibook.book1D("EtlHitEtaZposD2", "ETL SIM hits #eta (+Z, Second disk);#eta_{SIM}", 100, 1.56, 3.2);

  meHitTvsE_[0] =
      ibook.bookProfile("EtlHitTvsEZnegD1",
                        "ETL SIM time vs energy (-Z, Single(topo1D)/First(topo2D) disk);E_{SIM} [MeV];T_{SIM} [ns]",
                        50,
                        0.,
                        2.,
                        0.,
                        100.);
  meHitTvsE_[1] = ibook.bookProfile(
      "EtlHitTvsEZnegD2", "ETL SIM time vs energy (-Z, Second disk);E_{SIM} [MeV];T_{SIM} [ns]", 50, 0., 2., 0., 100.);
  meHitTvsE_[2] =
      ibook.bookProfile("EtlHitTvsEZposD1",
                        "ETL SIM time vs energy (+Z, Single(topo1D)/First(topo2D) disk);E_{SIM} [MeV];T_{SIM} [ns]",
                        50,
                        0.,
                        2.,
                        0.,
                        100.);
  meHitTvsE_[3] = ibook.bookProfile(
      "EtlHitTvsEZposD2", "ETL SIM time vs energy (+Z, Second disk);E_{SIM} [MeV];T_{SIM} [ns]", 50, 0., 2., 0., 100.);
  meHitEvsPhi_[0] =
      ibook.bookProfile("EtlHitEvsPhiZnegD1",
                        "ETL SIM energy vs #phi (-Z, Single(topo1D)/First(topo2D) disk);#phi_{SIM} [rad];E_{SIM} [MeV]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        100.);
  meHitEvsPhi_[1] = ibook.bookProfile("EtlHitEvsPhiZnegD2",
                                      "ETL SIM energy vs #phi (-Z, Second disk);#phi_{SIM} [rad];E_{SIM} [MeV]",
                                      50,
                                      -3.15,
                                      3.15,
                                      0.,
                                      100.);
  meHitEvsPhi_[2] =
      ibook.bookProfile("EtlHitEvsPhiZposD1",
                        "ETL SIM energy vs #phi (+Z, Single(topo1D)/First(topo2D) disk);#phi_{SIM} [rad];E_{SIM} [MeV]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        100.);
  meHitEvsPhi_[3] = ibook.bookProfile("EtlHitEvsPhiZposD2",
                                      "ETL SIM energy vs #phi (+Z, Second disk);#phi_{SIM} [rad];E_{SIM} [MeV]",
                                      50,
                                      -3.15,
                                      3.15,
                                      0.,
                                      100.);
  meHitEvsEta_[0] =
      ibook.bookProfile("EtlHitEvsEtaZnegD1",
                        "ETL SIM energy vs #eta (-Z, Single(topo1D)/First(topo2D) disk);#eta_{SIM};E_{SIM} [MeV]",
                        50,
                        -3.2,
                        -1.56,
                        0.,
                        100.);
  meHitEvsEta_[1] = ibook.bookProfile("EtlHitEvsEtaZnegD2",
                                      "ETL SIM energy vs #eta (-Z, Second disk);#eta_{SIM};E_{SIM} [MeV]",
                                      50,
                                      -3.2,
                                      -1.56,
                                      0.,
                                      100.);
  meHitEvsEta_[2] =
      ibook.bookProfile("EtlHitEvsEtaZposD1",
                        "ETL SIM energy vs #eta (+Z, Single(topo1D)/First(topo2D) disk);#eta_{SIM};E_{SIM} [MeV]",
                        50,
                        1.56,
                        3.2,
                        0.,
                        100.);
  meHitEvsEta_[3] = ibook.bookProfile("EtlHitEvsEtaZposD2",
                                      "ETL SIM energy vs #eta (+Z, Second disk);#eta_{SIM};E_{SIM} [MeV]",
                                      50,
                                      1.56,
                                      3.2,
                                      0.,
                                      100.);
  meHitTvsPhi_[0] =
      ibook.bookProfile("EtlHitTvsPhiZnegD1",
                        "ETL SIM time vs #phi (-Z, Single(topo1D)/First(topo2D) disk);#phi_{SIM} [rad];T_{SIM} [ns]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        100.);
  meHitTvsPhi_[1] = ibook.bookProfile("EtlHitTvsPhiZnegD2",
                                      "ETL SIM time vs #phi (-Z, Second disk);#phi_{SIM} [rad];T_{SIM} [ns]",
                                      50,
                                      -3.15,
                                      3.15,
                                      0.,
                                      100.);
  meHitTvsPhi_[2] =
      ibook.bookProfile("EtlHitTvsPhiZposD1",
                        "ETL SIM time vs #phi (+Z, Single(topo1D)/First(topo2D) disk);#phi_{SIM} [rad];T_{SIM} [ns]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        100.);
  meHitTvsPhi_[3] = ibook.bookProfile("EtlHitTvsPhiZposD2",
                                      "ETL SIM time vs #phi (+Z, Second disk);#phi_{SIM} [rad];T_{SIM} [ns]",
                                      50,
                                      -3.15,
                                      3.15,
                                      0.,
                                      100.);
  meHitTvsEta_[0] =
      ibook.bookProfile("EtlHitTvsEtaZnegD1",
                        "ETL SIM time vs #eta (-Z, Single(topo1D)/First(topo2D) disk);#eta_{SIM};T_{SIM} [ns]",
                        50,
                        -3.2,
                        -1.56,
                        0.,
                        100.);
  meHitTvsEta_[1] = ibook.bookProfile(
      "EtlHitTvsEtaZnegD2", "ETL SIM time vs #eta (-Z, Second disk);#eta_{SIM};T_{SIM} [ns]", 50, -3.2, -1.56, 0., 100.);
  meHitTvsEta_[2] =
      ibook.bookProfile("EtlHitTvsEtaZposD1",
                        "ETL SIM time vs #eta (+Z, Single(topo1D)/First(topo2D) disk);#eta_{SIM};T_{SIM} [ns]",
                        50,
                        1.56,
                        3.2,
                        0.,
                        100.);
  meHitTvsEta_[3] = ibook.bookProfile(
      "EtlHitTvsEtaZposD2", "ETL SIM time vs #eta (+Z, Second disk);#eta_{SIM};T_{SIM} [ns]", 50, 1.56, 3.2, 0., 100.);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void EtlSimHitsValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/ETL/SimHits");
  desc.add<edm::InputTag>("inputTag", edm::InputTag("mix", "g4SimHitsFastTimerHitsEndcap"));
  desc.add<double>("hitMinimumEnergy1Dis", 0.1);    // [MeV]
  desc.add<double>("hitMinimumEnergy2Dis", 0.001);  // [MeV]

  descriptions.add("etlSimHits", desc);
}

DEFINE_FWK_MODULE(EtlSimHitsValidation);
