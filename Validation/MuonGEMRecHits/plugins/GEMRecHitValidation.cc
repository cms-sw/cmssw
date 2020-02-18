#include "Validation/MuonGEMRecHits/plugins/GEMRecHitValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"


GEMRecHitValidation::GEMRecHitValidation(const edm::ParameterSet& pset)
    : GEMBaseValidation(pset, "GEMRecHitsValidation") {

  const auto& rechit_pset = pset.getParameterSet("gemRecHit");
  const auto& rechit_tag = rechit_pset.getParameter<edm::InputTag>("inputTag");
  rechit_token_ = consumes<GEMRecHitCollection>(rechit_tag);

  const auto& simhit_pset = pset.getParameterSet("gemSimHit");
  const auto& simhit_tag = simhit_pset.getParameter<edm::InputTag>("inputTag");
  simhit_token_ = consumes<edm::PSimHitContainer>(simhit_tag);
}


GEMRecHitValidation::~GEMRecHitValidation() {}


void GEMRecHitValidation::bookHistograms(DQMStore::IBooker& booker,
                                         edm::Run const& run,
                                         edm::EventSetup const& setup) {
  const GEMGeometry* gem = initGeometry(setup);

  // NOTE Cluster Size
  booker.setCurrentFolder("MuonGEMRecHitsV/GEMRecHitsTask/ClusterSize");

  TString cls_title = "Cluster Size Distribution";
  TString cls_x_title = "The number of adjacent strips";

  me_cls_ = booker.book1D("cls",
                          cls_title + ";" + cls_x_title + ";" + "Entries",
                          11, -0.5, 10.5);

  if (detail_plot_) {
    for (const auto& region : gem->regions()) {
      Int_t region_id = region->region();

      for (const auto& station : region->stations()) {
        Int_t station_id = station->station();

        const GEMSuperChamber* super_chamber = station->superChambers().front();
        for (const auto& chamber : super_chamber->chambers()) {
          Int_t layer_id = chamber->id().layer();
          ME3IdsKey key3(region_id, station_id, layer_id);

          me_detail_cls_[key3] = bookHist1D(
              booker, key3, "cls", cls_title, 11, -0.5, 10.5, cls_x_title);

        } // chamber loop
      } // station loop
    } // region loop
  } // detail plot

  // NOTE Residual
  booker.setCurrentFolder("MuonGEMRecHitsV/GEMRecHitsTask/Residual");

  for (const auto& region : gem->regions()) {
    Int_t region_id = region->region();

    me_residual_x_[region_id] = bookHist1D(
        booker, region_id, "residual_x", "Residual in X",
        120, -3, 3, "Residual in X [cm]");

    me_residual_y_[region_id] = bookHist1D(
        booker, region_id, "residual_y", "Residual in Y",
        600, -15, 15, "Residual in Y [cm]");

    if (detail_plot_) {
      for (const auto& station : region->stations()) {
        Int_t station_id = station->station();

        const GEMSuperChamber* super_chamber = station->superChambers().front();
        for (const auto& chamber : super_chamber->chambers()) {
          Int_t layer_id = chamber->id().layer();
          ME3IdsKey key3(region_id, station_id, layer_id);

          // Occupancy histograms of SimHits and RecHits for Efficiency
          me_detail_residual_x_[key3] = bookHist1D(
              booker, key3, "residual_x", "Residual in X",
              120, -3, 3, "Residual in X [cm]");

          me_detail_residual_y_[key3] = bookHist1D(
              booker, key3, "residual_y", "Residual in Y",
              600, -15, 15, "Residual in Y [cm]");

        } // chamber loop
      } // station loop
    } // detail_plot
  } // region loop

  // NOTE Pull
  booker.setCurrentFolder("MuonGEMRecHitsV/GEMRecHitsTask/Pull");

  for (const auto& region : gem->regions()) {
    Int_t region_id = region->region();

    me_pull_x_[region_id] = bookHist1D(
        booker, region_id, "pull_x", "Pull in X", 100, -3, 3);

    me_pull_y_[region_id] = bookHist1D(
        booker, region_id, "pull_y", "Pull in Y", 100, -3, 3);

    if (detail_plot_) {
      for (const auto& station : region->stations()) {
        Int_t station_id = station->station();

        const GEMSuperChamber* super_chamber = station->superChambers().front();
        for (const auto& chamber : super_chamber->chambers()) {
          Int_t layer_id = chamber->id().layer();
          ME3IdsKey key3(region_id, station_id, layer_id);

          me_detail_pull_x_[key3] = bookHist1D(
              booker, key3, "pull_x", "Pull in X", 100, -3, 3);

          me_detail_pull_y_[key3] = bookHist1D(
              booker, key3, "pull_y", "Pull in Y", 100, -3, 3);

        } // chamber loop
      } // station loop
    } // detail plot
  } // region loop

  // NOTE Occupancy
  booker.setCurrentFolder("MuonGEMRecHitsV/GEMRecHitsTask/Occupancy");
  for (const auto& region : gem->regions()) {
    Int_t region_id = region->region();

    me_simhit_occ_eta_[region_id] = bookHist1D(
        booker, region_id, "muon_simhit_occ_eta", "Muon SimHit Eta Occupancy",
        50, eta_range_[0], eta_range_[1], "|#eta|");

    me_rechit_occ_eta_[region_id] = bookHist1D(
        booker, region_id, "matched_rechit_occ_eta", "Matched RecHit Eta Occupancy",
        50, eta_range_[0], eta_range_[1], "|#eta|");

    for (const auto& station : region->stations()) {
      Int_t station_id = station->station();
      ME2IdsKey key2(region_id, station_id);

      me_simhit_occ_phi_[key2] = bookHist1D(
          booker, key2, "muon_simhit_occ_phi", "Muon SimHit Phi Occupancy",
          51, -M_PI, M_PI, "#phi");

      me_rechit_occ_phi_[key2] = bookHist1D(
          booker, key2, "matched_rechit_occ_phi", "Matched RecHit Phi Occupancy",
          51, -M_PI, M_PI, "#phi");

      me_simhit_occ_det_[key2] = bookDetectorOccupancy(
          booker, key2, station, "muon_simhit", "Muon SimHit");

      me_rechit_occ_det_[key2] = bookDetectorOccupancy(
          booker, key2, station, "matched_rechit", "Matched RecHit");
    }  // station loop
  }    // region_loop
}


Bool_t GEMRecHitValidation::matchRecHitAgainstSimHit(
    GEMRecHitCollection::const_iterator rechit, Int_t simhit_strip) {
  Bool_t matched = false;

  Int_t cls = rechit->clusterSize();
  Int_t rechit_first_strip = rechit->firstClusterStrip();

  if (cls == 1) {
    matched = simhit_strip == rechit_first_strip;
  } else {
    Int_t rechit_last_strip = rechit_first_strip + cls - 1;
    matched = (simhit_strip >= rechit_first_strip) and (simhit_strip <= rechit_last_strip);
  }

  return matched;
}


void GEMRecHitValidation::analyze(const edm::Event& event,
                                  const edm::EventSetup& setup) {
  const GEMGeometry* gem = initGeometry(setup);

  edm::Handle<edm::PSimHitContainer> simhit_container;
  event.getByToken(simhit_token_, simhit_container);
  if (not simhit_container.isValid()) {
    edm::LogError(kLogCategory_) << "Failed to get PSimHitContainer." << std::endl;
    return;
  }

  edm::Handle<GEMRecHitCollection> rechit_collection;
  event.getByToken(rechit_token_, rechit_collection);
  if (not rechit_collection.isValid()) {
    edm::LogError(kLogCategory_) << "Failed to get GEMRecHitCollection" << std::endl;
    return;
  }

  for (const auto& simhit : *simhit_container.product()) {
    if (gem->idToDet(simhit.detUnitId()) == nullptr) {
      edm::LogError(kLogCategory_) << "MuonGEMHit didn't matched with GEMGeometry." << std::endl;
      continue;
    }

    if (not isMuonSimHit(simhit))
      continue;

    GEMDetId simhit_gemid(simhit.detUnitId());
    const BoundPlane& surface = gem->idToDet(simhit_gemid)->surface();

    Int_t region_id = simhit_gemid.region();
    Int_t station_id = simhit_gemid.station();
    Int_t layer_id = simhit_gemid.layer();
    Int_t chamber_id = simhit_gemid.chamber();
    Int_t roll_id = simhit_gemid.roll();

    ME2IdsKey key2(region_id, station_id);
    ME3IdsKey key3(region_id, station_id, layer_id);

    const LocalPoint& simhit_local_pos = simhit.localPosition();
    const GlobalPoint& simhit_global_pos = surface.toGlobal(simhit_local_pos);

    Float_t simhit_g_abs_eta = std::fabs(simhit_global_pos.eta());
    Float_t simhit_g_phi = simhit_global_pos.phi();

    Int_t simhit_strip = gem->etaPartition(simhit_gemid)->strip(simhit_local_pos);
    Int_t det_occ_bin_x = getDetOccBinX(chamber_id, layer_id);

    me_simhit_occ_eta_[region_id]->Fill(simhit_g_abs_eta);
    me_simhit_occ_phi_[key2]->Fill(simhit_g_phi);
    me_simhit_occ_det_[key2]->Fill(det_occ_bin_x, roll_id);

    GEMRecHitCollection::range range = rechit_collection->get(simhit_gemid);
    for (auto rechit = range.first; rechit != range.second; ++rechit) {
      if (gem->idToDet(rechit->gemId()) == nullptr) {
        edm::LogError(kLogCategory_) << "GEMRecHit didn't matched with GEMGeometry." << std::endl;
        continue;
      }

      if (matchRecHitAgainstSimHit(rechit, simhit_strip)) {
        Int_t cls = rechit->clusterSize();

        const LocalPoint& rechit_local_pos = rechit->localPosition();

        Float_t resolution_x = std::sqrt(rechit->localPositionError().xx());
        Float_t resolution_y = std::sqrt(rechit->localPositionError().yy());

        Float_t residual_x = rechit_local_pos.x() - simhit_local_pos.x();
        Float_t residual_y = rechit_local_pos.y() - simhit_local_pos.y();

        Float_t pull_x = residual_x / resolution_x;
        Float_t pull_y = residual_y / resolution_y;

        me_cls_->Fill(cls);
        me_residual_x_[region_id]->Fill(residual_x);
        me_residual_y_[region_id]->Fill(residual_y);
        me_pull_x_[region_id]->Fill(pull_x);
        me_pull_y_[region_id]->Fill(pull_y);

        // NOTE If we use global position of rechit,
        // 'inconsistent bin contents' exception may occur.
        me_rechit_occ_eta_[region_id]->Fill(simhit_g_abs_eta);
        me_rechit_occ_phi_[key2]->Fill(simhit_g_phi);
        me_rechit_occ_det_[key2]->Fill(det_occ_bin_x, roll_id);

        if (detail_plot_) {
          me_detail_cls_[key3]->Fill(cls);

          me_detail_residual_x_[key3]->Fill(residual_x);
          me_detail_residual_y_[key3]->Fill(residual_y);

          me_detail_pull_x_[key3]->Fill(pull_x);
          me_detail_pull_y_[key3]->Fill(pull_y);
        } // detail_plot

        // NOTE If we find GEMRecHit that matches PSimHit, then exit GEMRecHitCollection loop.
        break;

      } // if rechit matches against simhit 
    } // rechit loop
  } // simhit loop
}
