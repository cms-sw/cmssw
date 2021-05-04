#include "Validation/MuonGEMHits/plugins/GEMSimHitValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

GEMSimHitValidation::GEMSimHitValidation(const edm::ParameterSet& pset)
    : GEMBaseValidation(pset, "GEMSimHitValidation") {
  const auto& simhit_pset = pset.getParameterSet("gemSimHit");
  const auto& simhit_tag = simhit_pset.getParameter<edm::InputTag>("inputTag");
  simhit_token_ = consumes<edm::PSimHitContainer>(simhit_tag);

  tof_range_ = pset.getUntrackedParameter<std::vector<Double_t> >("TOFRange");
  geomToken_ = esConsumes<GEMGeometry, MuonGeometryRecord>();
  geomTokenBeginRun_ = esConsumes<GEMGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();
}

GEMSimHitValidation::~GEMSimHitValidation() {}

void GEMSimHitValidation::bookHistograms(DQMStore::IBooker& booker, edm::Run const& run, edm::EventSetup const& setup) {
  const GEMGeometry* gem = &setup.getData(geomTokenBeginRun_);

  // NOTE Time of flight
  booker.setCurrentFolder("MuonGEMHitsV/GEMHitsTask/TimeOfFlight");

  TString tof_xtitle = "Time of flight [ns]";
  TString tof_ytitle = "Entries";

  for (const auto& region : gem->regions()) {
    Int_t region_id = region->region();

    for (const auto& station : region->stations()) {
      Int_t station_id = station->station();

      const auto [tof_min, tof_max] = getTOFRange(station_id);
      ME2IdsKey key2{region_id, station_id};

      me_tof_mu_[key2] =
          bookHist1D(booker, key2, "tof_muon", "SimHit TOF (Muon only)", 20, tof_min, tof_max, tof_xtitle, tof_ytitle);

      me_tof_others_[key2] = bookHist1D(
          booker, key2, "tof_others", "SimHit TOF (Other Particles)", 20, tof_min, tof_max, tof_xtitle, tof_ytitle);
    }  // station loop
  }    // region loop

  if (detail_plot_) {
    for (const auto& region : gem->regions()) {
      Int_t region_id = region->region();

      for (const auto& station : region->stations()) {
        Int_t station_id = station->station();

        const auto [tof_min, tof_max] = getTOFRange(station_id);
        const auto& superChamberVec = station->superChambers();
        if (superChamberVec.empty() || superChamberVec.front() == nullptr) {
          edm::LogError(kLogCategory_) << "Super chambers missing or null for region = " << region_id
                                       << " and station = " << station_id;
        } else {
          const GEMSuperChamber* super_chamber = superChamberVec.front();

          for (const auto& chamber : super_chamber->chambers()) {
            Int_t layer_id = chamber->id().layer();
            ME3IdsKey key3{region_id, station_id, layer_id};

            me_detail_tof_[key3] = bookHist1D(
                booker, key3, "tof", "Time of Flight of Muon SimHits", 40, tof_min, tof_max, tof_xtitle, tof_ytitle);

            me_detail_tof_mu_[key3] = bookHist1D(
                booker, key3, "tof_muon", "SimHit TOF (Muon only)", 40, tof_min, tof_max, tof_xtitle, tof_ytitle);
          }  // chamber loop
        }    // end else
      }      // station loop
    }        // region loop
  }          // detail plot

  // NOTE Energy Loss
  booker.setCurrentFolder("MuonGEMHitsV/GEMHitsTask/EnergyLoss");

  TString eloss_xtitle = "Energy loss [eV]";
  TString eloss_ytitle = "Entries / 0.5 keV";

  for (const auto& station : gem->regions()[0]->stations()) {
    Int_t station_id = station->station();

    auto eloss_mu_name = TString::Format("eloss_muon_st%d", station_id);
    auto eloss_mu_title = TString::Format("SimHit Energy Loss (Muon only) : Station %d", station_id);

    me_eloss_mu_[station_id] =
        booker.book1D(eloss_mu_name, eloss_mu_title + ";" + eloss_xtitle + ";" + eloss_ytitle, 20, 0.0, 10.0);

    auto eloss_others_name = TString::Format("eloss_others_st%d", station_id);
    auto eloss_others_title = TString::Format("SimHit Energy Loss (Other Particles) : Station %d", station_id);

    me_eloss_others_[station_id] =
        booker.book1D(eloss_others_name, eloss_others_title + ";" + eloss_xtitle + ";" + eloss_ytitle, 20, 0.0, 10.0);
  }  // station loop

  if (detail_plot_) {
    for (const auto& region : gem->regions()) {
      Int_t region_id = region->region();
      for (const auto& station : region->stations()) {
        Int_t station_id = station->station();
        const auto& superChamberVec = station->superChambers();
        if (superChamberVec.empty() || superChamberVec.front() == nullptr) {
          edm::LogError(kLogCategory_) << "Super chambers missing or null for region = " << region_id
                                       << " and station = " << station_id;
        } else {
          for (const auto& chamber : superChamberVec.front()->chambers()) {
            Int_t layer_id = chamber->id().layer();
            ME3IdsKey key3{region_id, station_id, layer_id};

            me_detail_eloss_[key3] =
                bookHist1D(booker, key3, "eloss", "SimHit Energy Loss", 60, 0.0, 6000.0, eloss_xtitle, eloss_ytitle);

            me_detail_eloss_mu_[key3] = bookHist1D(booker,
                                                   key3,
                                                   "eloss_muon",
                                                   "SimHit Energy Loss (Muon Only)",
                                                   60,
                                                   0.0,
                                                   6000.0,
                                                   eloss_xtitle,
                                                   eloss_ytitle);

          }  // chamber loop
        }    // end else
      }      // station loop
    }        // region loop
  }          // detail plot

  // NOTE Occupancy
  booker.setCurrentFolder("MuonGEMHitsV/GEMHitsTask/Occupancy");

  for (const auto& region : gem->regions()) {
    Int_t region_id = region->region();

    if (detail_plot_)
      me_detail_occ_zr_[region_id] = bookZROccupancy(booker, region_id, "simhit", "SimHit");

    for (const auto& station : region->stations()) {
      Int_t station_id = station->station();
      ME2IdsKey key2{region_id, station_id};

      if (detail_plot_) {
        me_detail_occ_det_[key2] = bookDetectorOccupancy(booker, key2, station, "simhit", "SimHit");
        me_detail_occ_det_mu_[key2] = bookDetectorOccupancy(booker, key2, station, "muon_simhit", "Muon SimHit");
      }

      const auto& superChamberVec = station->superChambers();
      if (superChamberVec.empty() || superChamberVec.front() == nullptr) {
        edm::LogError(kLogCategory_) << "Super chambers missing or null for region = " << region_id
                                     << " and station = " << station_id;
      } else {
        const GEMSuperChamber* super_chamber = superChamberVec.front();
        for (const auto& chamber : super_chamber->chambers()) {
          Int_t layer_id = chamber->id().layer();
          ME3IdsKey key3{region_id, station_id, layer_id};

          me_occ_eta_mu_[key3] = bookHist1D(booker,
                                            key3,
                                            "muon_simhit_occ_eta",
                                            "Muon SimHit Eta Occupancy",
                                            16,
                                            eta_range_[station_id * 2 + 0],
                                            eta_range_[station_id * 2 + 1],
                                            "#eta");

          me_occ_phi_mu_[key3] = bookHist1D(
              booker, key3, "muon_simhit_occ_phi", "Muon SimHit Phi Occupancy", 36, -5, 355, "#phi [degrees]");

          me_occ_pid_[key3] = bookPIDHist(booker, key3, "simhit_occ_pid", "Number of entries for each paritcle");

          if (detail_plot_)
            me_detail_occ_xy_[key3] = bookXYOccupancy(booker, key3, "simhit", "SimHit");
        }  // layer loop
      }    // end else
    }      // station loop
  }        // region loop
}

std::tuple<Double_t, Double_t> GEMSimHitValidation::getTOFRange(Int_t station_id) {
  UInt_t start_index = station_id * 2;
  Double_t tof_min = tof_range_[start_index];
  Double_t tof_max = tof_range_[start_index + 1];
  return std::make_tuple(tof_min, tof_max);
}

void GEMSimHitValidation::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  const GEMGeometry* gem = &setup.getData(geomToken_);

  edm::Handle<edm::PSimHitContainer> simhit_container;
  event.getByToken(simhit_token_, simhit_container);
  if (not simhit_container.isValid()) {
    edm::LogError(kLogCategory_) << "Cannot get GEMHits by Token simhitLabel" << std::endl;
    return;
  }

  for (const auto& simhit : *simhit_container.product()) {
    const GEMDetId gemid(simhit.detUnitId());

    if (gem->idToDet(gemid) == nullptr) {
      edm::LogError(kLogCategory_) << "SimHit did not matched with GEM Geometry." << std::endl;
      continue;
    }

    Int_t region_id = gemid.region();
    Int_t station_id = gemid.station();
    Int_t layer_id = gemid.layer();
    Int_t chamber_id = gemid.chamber();
    Int_t ieta = gemid.ieta();
    Int_t num_layers = gemid.nlayers();

    ME2IdsKey key2{region_id, station_id};
    ME3IdsKey key3{region_id, station_id, layer_id};

    GlobalPoint&& simhit_global_pos = gem->idToDet(gemid)->surface().toGlobal(simhit.localPosition());

    Float_t simhit_g_r = simhit_global_pos.perp();
    Float_t simhit_g_x = simhit_global_pos.x();
    Float_t simhit_g_y = simhit_global_pos.y();
    Float_t simhit_g_abs_z = std::fabs(simhit_global_pos.z());
    Float_t simhit_g_eta = std::fabs(simhit_global_pos.eta());
    Float_t simhit_g_phi = toDegree(simhit_global_pos.phi());

    Float_t energy_loss = kEnergyCF_ * simhit.energyLoss();
    energy_loss = energy_loss > 10 ? 9.9 : energy_loss;
    Float_t tof = simhit.timeOfFlight();
    Int_t pid = simhit.particleType();
    Int_t pid_idx = getPidIdx(pid);

    // NOTE Fill MonitorElement
    Int_t bin_x = getDetOccBinX(num_layers, chamber_id, layer_id);

    bool is_muon_simhit = isMuonSimHit(simhit);
    if (is_muon_simhit) {
      me_tof_mu_[key2]->Fill(tof);
      me_eloss_mu_[station_id]->Fill(energy_loss);
      me_occ_eta_mu_[key3]->Fill(simhit_g_eta);
      me_occ_phi_mu_[key3]->Fill(simhit_g_phi);
    } else {
      me_tof_others_[key2]->Fill(tof);
      me_eloss_others_[station_id]->Fill(energy_loss);
    }

    me_occ_pid_[key3]->Fill(pid_idx);

    if (detail_plot_) {
      me_detail_tof_[key3]->Fill(tof);
      me_detail_eloss_[key3]->Fill(energy_loss);

      me_detail_occ_zr_[region_id]->Fill(simhit_g_abs_z, simhit_g_r);
      me_detail_occ_det_[key2]->Fill(bin_x, ieta);
      me_detail_occ_xy_[key3]->Fill(simhit_g_x, simhit_g_y);

      if (is_muon_simhit) {
        me_detail_tof_mu_[key3]->Fill(tof);
        me_detail_eloss_mu_[key3]->Fill(energy_loss);
        me_detail_occ_det_mu_[key2]->Fill(bin_x, ieta);
      }

    }  // detail_plot
  }    // simhit loop
}
