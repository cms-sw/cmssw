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

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"

#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

#include "MTDHit.h"

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

  static constexpr float BXTime_ = 25.;  // [ns]

  edm::EDGetTokenT<CrossingFrame<PSimHit> > btlSimHitsToken_;

  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;

  // --- histograms declaration

  MonitorElement* meNevents_;

  MonitorElement* meNhits_;
  MonitorElement* meNtrkPerCell_;

  MonitorElement* meHitEnergy_;
  MonitorElement* meHitLogEnergy_;
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

  MonitorElement* meHitTrkID1_;
  MonitorElement* meHitTrkID2_;
  MonitorElement* meHitTrkID3_;
  MonitorElement* meHitTrkID4_;

  MonitorElement* meHitDeltaT2_;
  MonitorElement* meHitDeltaT3_;
  MonitorElement* meHitDeltaT4_;

  MonitorElement* meHitDeltaE12vsE1_;

  static constexpr float cellEneCut_ = 10.;  // [MeV]

  MonitorElement* meHitE1overEcellBulk1_;
  MonitorElement* meHitE1overEcellBulk2_;
  MonitorElement* meHitE1overEcellBulk3_;
  MonitorElement* meHitE1overEcellBulk4_;
  MonitorElement* meHitE1overEcellTail1_;
  MonitorElement* meHitE1overEcellTail2_;
  MonitorElement* meHitE1overEcellTail3_;
  MonitorElement* meHitE1overEcellTail4_;
};

// ------------ constructor and destructor --------------
BtlSimHitsValidation::BtlSimHitsValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      hitMinEnergy_(iConfig.getParameter<double>("hitMinimumEnergy")) {
  btlSimHitsToken_ = consumes<CrossingFrame<PSimHit> >(iConfig.getParameter<edm::InputTag>("inputTag"));
  mtdgeoToken_ = esConsumes<MTDGeometry, MTDDigiGeometryRecord>();
  mtdtopoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>();
}

BtlSimHitsValidation::~BtlSimHitsValidation() {}

// ------------ method called for each event  ------------
void BtlSimHitsValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace geant_units::operators;

  auto geometryHandle = iSetup.getTransientHandle(mtdgeoToken_);
  const MTDGeometry* geom = geometryHandle.product();

  auto topologyHandle = iSetup.getTransientHandle(mtdtopoToken_);
  const MTDTopology* topology = topologyHandle.product();

  auto btlSimHitsHandle = makeValid(iEvent.getHandle(btlSimHitsToken_));
  MixCollection<PSimHit> btlSimHits(btlSimHitsHandle.product());

  std::unordered_map<uint32_t, std::unordered_map<uint64_t, MTDHit> > m_btlHitsPerCell;

  // --- Loop over the BLT SIM hits and accumulate the hits with the same track ID in each cell
  for (auto const& simHit : btlSimHits) {
    // --- Use only SIM hits compatible with the in-time bunch-crossing
    if (simHit.tof() < 0. || simHit.tof() > BXTime_) {
      continue;
    }

    DetId id = simHit.detUnitId();

    // --- Build a global track ID by combining the SIM hit's eventId and trackId
    uint64_t globalTrkID = ((uint64_t)simHit.eventId().rawId() << 32) | simHit.trackId();

    // --- Sum the energies of SIM hits with the same track ID in the same cell
    m_btlHitsPerCell[id.rawId()][globalTrkID].energy += convertGeVToMeV(simHit.energyLoss());

    // --- Assign the time and position of the first SIM hit in time to the accumulated hit
    if (m_btlHitsPerCell[id.rawId()][globalTrkID].time == 0. ||
        simHit.tof() < m_btlHitsPerCell[id.rawId()][globalTrkID].time) {
      m_btlHitsPerCell[id.rawId()][globalTrkID].time = simHit.tof();

      auto hit_pos = simHit.localPosition();
      m_btlHitsPerCell[id.rawId()][globalTrkID].x = hit_pos.x();
      m_btlHitsPerCell[id.rawId()][globalTrkID].y = hit_pos.y();
      m_btlHitsPerCell[id.rawId()][globalTrkID].z = hit_pos.z();
    }

  }  // simHit loop

  // ==============================================================================
  //  Histogram filling
  // ==============================================================================

  if (!m_btlHitsPerCell.empty()) {
    meNhits_->Fill(log10(m_btlHitsPerCell.size()));
  }

  // --- Loop over the BTL cells
  for (auto const& cell : m_btlHitsPerCell) {
    // --- Get the map of the hits in the cell
    const auto& m_hits = cell.second;

    // --- Skip cells with no hits
    if (m_hits.empty()) {
      continue;
    }

    // --- Loop over the hits in the cell, sum the hit energies and store the hit IDs in a vector
    std::vector<uint64_t> v_hitID;
    float ene_tot_cell = 0.;

    for (auto const& hit : m_hits) {
      ene_tot_cell += hit.second.energy;
      v_hitID.push_back(hit.first);
    }

    meHitLogEnergy_->Fill(log10(ene_tot_cell));

    // --- Skip cells with a total anergy less than hitMinEnergy_
    if (ene_tot_cell < hitMinEnergy_) {
      continue;
    }

    // --- Order the hits in time
    bool swapped;
    for (unsigned int ihit = 0; ihit < v_hitID.size() - 1; ++ihit) {
      swapped = false;
      for (unsigned int jhit = 0; jhit < v_hitID.size() - ihit - 1; ++jhit) {
        if (m_hits.at(v_hitID[jhit]).time > m_hits.at(v_hitID[jhit + 1]).time) {
          std::swap(v_hitID[jhit], v_hitID[jhit + 1]);
          swapped = true;
        }
      }
      if (swapped == false) {
        break;
      }
    }

    // --- Get the longer time interval between the hits in the cell
    float deltaT_max = 0.;
    for (unsigned int ihit = 0; ihit < v_hitID.size() - 1; ++ihit) {
      float deltaT = m_hits.at(v_hitID[ihit + 1]).time - m_hits.at(v_hitID[ihit]).time;

      if (deltaT > deltaT_max) {
        deltaT_max = deltaT;
      }
    }

    // --- Get the hit global position
    BTLDetId detId(cell.first);
    DetId geoId = detId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode()));
    const MTDGeomDet* thedet = geom->idToDet(geoId);
    if (thedet == nullptr) {
      throw cms::Exception("BtlSimHitsValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                   << detId.rawId() << ") is invalid!" << std::dec << std::endl;
    }
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

    Local3DPoint local_point(convertMmToCm(m_hits.at(v_hitID[0]).x),
                             convertMmToCm(m_hits.at(v_hitID[0]).y),
                             convertMmToCm(m_hits.at(v_hitID[0]).z));

    local_point = topo.pixelToModuleLocalPoint(local_point, detId.row(topo.nrows()), detId.column(topo.nrows()));
    const auto& global_point = thedet->toGlobal(local_point);

    // --- Fill the histograms

    meHitEnergy_->Fill(ene_tot_cell);
    meHitTime_->Fill(m_hits.at(v_hitID[0]).time);

    meHitXlocal_->Fill(m_hits.at(v_hitID[0]).x);
    meHitYlocal_->Fill(m_hits.at(v_hitID[0]).y);
    meHitZlocal_->Fill(m_hits.at(v_hitID[0]).z);

    meOccupancy_->Fill(global_point.z(), global_point.phi());

    meHitX_->Fill(global_point.x());
    meHitY_->Fill(global_point.y());
    meHitZ_->Fill(global_point.z());
    meHitPhi_->Fill(global_point.phi());
    meHitEta_->Fill(global_point.eta());

    meHitTvsE_->Fill(ene_tot_cell, m_hits.at(v_hitID[0]).time);
    meHitEvsPhi_->Fill(global_point.phi(), ene_tot_cell);
    meHitEvsEta_->Fill(global_point.eta(), ene_tot_cell);
    meHitEvsZ_->Fill(global_point.z(), ene_tot_cell);
    meHitTvsPhi_->Fill(global_point.phi(), m_hits.at(v_hitID[0]).time);
    meHitTvsEta_->Fill(global_point.eta(), m_hits.at(v_hitID[0]).time);
    meHitTvsZ_->Fill(global_point.z(), m_hits.at(v_hitID[0]).time);

    meNtrkPerCell_->Fill(v_hitID.size());

    // --- First hit in the cell
    int trackID = (int)(v_hitID[0] & 0xFFFFFFFF) / 100000000;
    meHitTrkID1_->Fill(trackID);

    // cells in the bulk of energy distribution
    if (ene_tot_cell < cellEneCut_) {
      meHitE1overEcellBulk1_->Fill(m_hits.at(v_hitID[0]).energy / ene_tot_cell);
    }
    // cells in the tail of energy distribution
    else {
      meHitE1overEcellTail1_->Fill(m_hits.at(v_hitID[0]).energy / ene_tot_cell);
    }

    // --- Second hit in the cell
    if (v_hitID.size() == 2) {
      trackID = (int)(v_hitID[1] & 0xFFFFFFFF) / 100000000;
      meHitTrkID2_->Fill(trackID);

      meHitDeltaT2_->Fill(deltaT_max);

      // cells in the bulk of energy distribution
      if (ene_tot_cell < cellEneCut_) {
        meHitE1overEcellBulk2_->Fill(m_hits.at(v_hitID[1]).energy / ene_tot_cell);
      }
      // cells in the tail of energy distribution
      else {
        meHitE1overEcellTail2_->Fill(m_hits.at(v_hitID[1]).energy / ene_tot_cell);
      }

      meHitDeltaE12vsE1_->Fill(m_hits.at(v_hitID[0]).energy,
                               m_hits.at(v_hitID[0]).energy - m_hits.at(v_hitID[1]).energy);

    }
    // --- Third hit in the cell
    else if (v_hitID.size() == 3) {
      trackID = (int)(v_hitID[2] & 0xFFFFFFFF) / 100000000;
      meHitTrkID3_->Fill(trackID);

      meHitDeltaT3_->Fill(deltaT_max);

      // cells in the bulk of energy distribution
      if (ene_tot_cell < cellEneCut_) {
        meHitE1overEcellBulk3_->Fill(m_hits.at(v_hitID[2]).energy / ene_tot_cell);
      }
      // cells in the tail of energy distribution
      else {
        meHitE1overEcellTail3_->Fill(m_hits.at(v_hitID[2]).energy / ene_tot_cell);
      }

    }
    // --- Fourth hit in the cell and next ones
    else if (v_hitID.size() >= 4) {
      for (unsigned int ihit = 3; ihit < v_hitID.size(); ++ihit) {
        trackID = (int)(v_hitID[ihit] & 0xFFFFFFFF) / 100000000;
        meHitTrkID4_->Fill(trackID);

        // cells in the bulk of energy distribution
        if (ene_tot_cell < cellEneCut_) {
          meHitE1overEcellBulk4_->Fill(m_hits.at(v_hitID[ihit]).energy / ene_tot_cell);
        }
        // cells in the tail of energy distribution
        else {
          meHitE1overEcellTail4_->Fill(m_hits.at(v_hitID[ihit]).energy / ene_tot_cell);
        }
      }

      meHitDeltaT4_->Fill(deltaT_max);
    }

  }  // cell loop

  // --- This is to count the number of processed events, needed in the harvesting step
  meNevents_->Fill(0.5);
}

// ------------ method for histogram booking ------------
void BtlSimHitsValidation::bookHistograms(DQMStore::IBooker& ibook,
                                          edm::Run const& run,
                                          edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  // --- histograms booking

  meNevents_ = ibook.book1D("BtlNevents", "Number of events", 1, 0., 1.);

  meNhits_ = ibook.book1D("BtlNhits", "Number of BTL cells with SIM hits;log_{10}(N_{BTL cells})", 100, 0., 5.25);
  meNtrkPerCell_ = ibook.book1D("BtlNtrkPerCell", "Number of tracks per BTL cell;N_{trk}", 10, 0., 10.);

  meHitEnergy_ = ibook.book1D("BtlHitEnergy", "BTL SIM hits energy;E_{SIM} [MeV]", 100, 0., 20.);
  meHitLogEnergy_ = ibook.book1D("BtlHitLogEnergy", "BTL SIM hits energy;log_{10}(E_{SIM} [MeV])", 200, -6., 3.);
  meHitTime_ = ibook.book1D("BtlHitTime", "BTL SIM hits ToA;ToA_{SIM} [ns]", 100, 0., 25.);

  meHitXlocal_ = ibook.book1D("BtlHitXlocal", "BTL SIM local X;X_{SIM}^{LOC} [mm]", 100, -30., 30.);
  meHitYlocal_ = ibook.book1D("BtlHitYlocal", "BTL SIM local Y;Y_{SIM}^{LOC} [mm]", 100, -1.65, 1.65);
  meHitZlocal_ = ibook.book1D("BtlHitZlocal", "BTL SIM local Z;Z_{SIM}^{LOC} [mm]", 100, -2., 2.);

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

  meHitTrkID1_ = ibook.book1I("BtlHitTrkID1", "Category of the 1^{st} hit in time;Hit category", 8, 0, 8);
  meHitTrkID2_ = ibook.book1I("BtlHitTrkID2", "Category of the 2^{nd} hit in time;Hit category", 8, 0, 8);
  meHitTrkID3_ = ibook.book1I("BtlHitTrkID3", "Category of the 3^{rd} hit in time;Hit category", 8, 0, 8);
  meHitTrkID4_ = ibook.book1I("BtlHitTrkID4", "Category of the #geq4^{th} hit in time;Hit category", 8, 0, 8);

  meHitDeltaT2_ = ibook.book1D(
      "BtlHitDeltaT2", "Time interval between hits in the same cell (2 hits);#DeltaT_{2} [ns]", 100., 0., 25.);
  meHitDeltaT3_ = ibook.book1D(
      "BtlHitDeltaT3", "Max time interval between hits in the same cell (3 hits);#DeltaT_{3} [ns]", 100., 0., 25.);
  meHitDeltaT4_ = ibook.book1D("BtlHitDeltaT4",
                               "Max time interval between hits in the same cell (#geq4 hits);#DeltaT_{#geq4} [ns]",
                               100.,
                               0.,
                               25.);

  meHitDeltaE12vsE1_ = ibook.bookProfile(
      "BtlHitDeltaE12vsE", "E_{1}-E_{2} vs E_{1};E_{1} [MeV];E_{1}-E_{2} [MeV]", 100, 0., 100., 0., 100.);

  meHitE1overEcellBulk1_ = ibook.book1D("BtlHitE1overEtotBulk1",
                                        "Energy fraction of the 1^{st} hit in time (E_{cell}<10 MeV);E_{1} / E_{cell}",
                                        100,
                                        0.,
                                        1.);
  meHitE1overEcellBulk2_ = ibook.book1D("BtlHitE1overEtotBulk2",
                                        "Energy fraction of the 2^{nd} hit in time (E_{cell}<10 MeV);E_{2} / E_{cell}",
                                        100,
                                        0.,
                                        1.);
  meHitE1overEcellBulk3_ = ibook.book1D("BtlHitE1overEtotBulk3",
                                        "Energy fraction of the 3^{rd} hit in time (E_{cell}<10 MeV);E_{3} / E_{cell}",
                                        100,
                                        0.,
                                        1.);
  meHitE1overEcellBulk4_ =
      ibook.book1D("BtlHitE1overEtotBulk4",
                   "Energy fraction of the #geq4^{th} hit in time (E_{cell}<10 MeV);E_{#geq4} / E_{cell}",
                   100,
                   0.,
                   1.);
  meHitE1overEcellTail1_ = ibook.book1D("BtlHitE1overEtotTail1",
                                        "Energy fraction of the 1^{st} hit in time (E_{cell}>10 MeV);E_{1} / E_{cell}",
                                        100,
                                        0.,
                                        1.);
  meHitE1overEcellTail2_ = ibook.book1D("BtlHitE1overEtotTail2",
                                        "Energy fraction of the 2^{nd} hit in time (E_{cell}>10 MeV);E_{2} / E_{cell}",
                                        100,
                                        0.,
                                        1.);
  meHitE1overEcellTail3_ = ibook.book1D("BtlHitE1overEtotTail3",
                                        "Energy fraction of the 3^{rd} hit in time (E_{cell}>10 MeV);E_{3} / E_{cell}",
                                        100,
                                        0.,
                                        1.);
  meHitE1overEcellTail4_ =
      ibook.book1D("BtlHitE1overEtotTail4",
                   "Energy fraction of the #geq4^{th} hit in time (E_{cell}>10 MeV);E_{#geq4} / E_{cell}",
                   100,
                   0.,
                   1.);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BtlSimHitsValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/BTL/SimHits");
  desc.add<edm::InputTag>("inputTag", edm::InputTag("mix", "g4SimHitsFastTimerHitsBarrel"));
  desc.add<double>("hitMinimumEnergy", 1.);  // [MeV]

  descriptions.add("btlSimHitsValid", desc);
}

DEFINE_FWK_MODULE(BtlSimHitsValidation);
