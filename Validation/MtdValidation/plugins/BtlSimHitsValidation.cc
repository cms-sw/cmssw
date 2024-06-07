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
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

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

  MonitorElement* meHitDeltaT_;
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
    if (simHit.tof() < 0 || simHit.tof() > 25.)
      continue;

    DetId id = simHit.detUnitId();

    // --- Build a global track ID by combining the SIM hit's eventId and trackId
    uint64_t globalTrkID = ((uint64_t)simHit.eventId().rawId() << 32) | simHit.trackId();

    // --- Sum the energies of SIM hits with the same track ID in the same cell
    m_btlHitsPerCell[id.rawId()][globalTrkID].energy += convertUnitsTo(0.001_MeV, simHit.energyLoss());

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

  if (!m_btlHitsPerCell.empty())
    meNhits_->Fill(log10(m_btlHitsPerCell.size()));

  // --- Loop over the BTL cells
  for (auto const& cell : m_btlHitsPerCell) {
    // --- Get the map of the hits in the cell
    const auto& m_hits = cell.second;

    // --- Skip cells with no hits
    if (m_hits.empty())
      continue;

    // --- Loop over the hits in the cell, sum the hit energies and store the hit IDs in a vector
    std::vector<uint64_t> v_hitID;
    float ene_tot_cell = 0.;

    for (auto const& hit : m_hits) {
      ene_tot_cell += hit.second.energy;
      v_hitID.push_back(hit.first);
    }

    meHitLogEnergy_->Fill(log10(ene_tot_cell));

    // --- Skip cells with a total anergy less than hitMinEnergy_
    if (ene_tot_cell < hitMinEnergy_)
      continue;

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
      if (swapped == false)
        break;
    }

    // --- Get the hit global position

    BTLDetId detId(cell.first);
    DetId geoId = detId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode()));
    const MTDGeomDet* thedet = geom->idToDet(geoId);
    if (thedet == nullptr)
      throw cms::Exception("BtlSimHitsValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                   << detId.rawId() << ") is invalid!" << std::dec << std::endl;
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

    // first hit in the cell
    int trackID = (int)(v_hitID[0] & 0xFFFFFFFF) / 100000000;
    meHitTrkID1_->Fill(trackID);

    // second hit in the cell
    if (v_hitID.size() == 2) {
      float deltaT = m_hits.at(v_hitID[1]).time - m_hits.at(v_hitID[0]).time;
      meHitDeltaT_->Fill(deltaT);

      trackID = (int)(v_hitID[1] & 0xFFFFFFFF) / 100000000;
      meHitTrkID2_->Fill(trackID);

    }
    // third hit in the cell
    else if (v_hitID.size() == 3) {
      float deltaT = std::max(m_hits.at(v_hitID[1]).time - m_hits.at(v_hitID[0]).time,
                              m_hits.at(v_hitID[2]).time - m_hits.at(v_hitID[1]).time);
      meHitDeltaT_->Fill(deltaT);

      trackID = (int)(v_hitID[2] & 0xFFFFFFFF) / 100000000;
      meHitTrkID3_->Fill(trackID);

    }
    // fourth hit in the cell
    else if (v_hitID.size() == 4) {
      float deltaT = std::max(std::max(m_hits.at(v_hitID[1]).time - m_hits.at(v_hitID[0]).time,
                                       m_hits.at(v_hitID[2]).time - m_hits.at(v_hitID[1]).time),
                              m_hits.at(v_hitID[3]).time - m_hits.at(v_hitID[2]).time);
      meHitDeltaT_->Fill(deltaT);

      trackID = (int)(v_hitID[3] & 0xFFFFFFFF) / 100000000;
      meHitTrkID4_->Fill(trackID);
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

  meHitTrkID1_ = ibook.book1I("BtlHitTrkID1", "Track ID of the first hit;trackID", 10, 0, 10);
  meHitTrkID2_ = ibook.book1I("BtlHitTrkID2", "Track ID of the second hit;trackID", 10, 0, 10);
  meHitTrkID3_ = ibook.book1I("BtlHitTrkID3", "Track ID of the third hit;trackID", 10, 0, 10);
  meHitTrkID4_ = ibook.book1I("BtlHitTrkID4", "Track ID of the fourth hit;trackID", 10, 0, 10);

  meHitDeltaT_ = ibook.book1D("BtlHitDeltaT", "Time difference between hits in the same cell", 100., 0., 25.);
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
