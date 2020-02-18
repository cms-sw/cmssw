#include "Validation/MuonGEMHits/plugins/GEMSimHitValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"


GEMSimHitValidation::GEMSimHitValidation(const edm::ParameterSet& pset)
    : GEMBaseValidation(pset, "GEMSimHitValidation") {
  const auto& simhit_pset = pset.getParameterSet("gemSimHit");
  const auto& simhit_tag = simhit_pset.getParameter<edm::InputTag>("inputTag");
  simhit_token_ = consumes<edm::PSimHitContainer>(simhit_tag);

  tof_range_ = pset.getUntrackedParameter<std::vector<Double_t> >("TOFRange");
}


GEMSimHitValidation::~GEMSimHitValidation() {}


void GEMSimHitValidation::bookHistograms(DQMStore::IBooker& booker,
                                         edm::Run const& run,
                                         edm::EventSetup const& setup) {
  const GEMGeometry* gem = initGeometry(setup);

  // NOTE Time of flight
  booker.setCurrentFolder("MuonGEMHitsV/GEMHitsTask/TimeOfFlight");

  for (const auto& station : gem->regions()[0]->stations()) {
    Int_t station_id = station->station();
    const auto [tof_min, tof_max] = getTOFRange(station_id);
    const char* tof_name = TString::Format("tof_muon_st%d", station_id);
    const char* tof_title = TString::Format("SimHit Time Of Flight (Muon only) : Station %d;Time of flight [ns];Entries", station_id);

    me_tof_mu_[station_id] = booker.book1D(tof_name, tof_title, 40, tof_min, tof_max);
  }

  if (detail_plot_) {
    for (const auto& region : gem->regions()) {
      Int_t region_id = region->region();

      for (const auto& station : region->stations()) {
        Int_t station_id = station->station();

        const auto [tof_min, tof_max] = getTOFRange(station_id);
        const GEMSuperChamber* super_chamber = station->superChambers().front();

        for (const auto& chamber : super_chamber->chambers()) {
          Int_t layer_id = chamber->id().layer();
          ME3IdsKey key3(region_id, station_id, layer_id);

          me_detail_tof_[key3] = bookHist1D(
              booker, key3, "tof", "Time of Flight of Muon SimHits",
              40, tof_min, tof_max, "Time of flight [ns]");

          me_detail_tof_mu_[key3] = bookHist1D(
              booker, key3, "tof_muon", "SimHit TOF (Muon only)",
              40, tof_min, tof_max, "Time of flight [ns]");

        } // chamber loop
      } // station loop
    } // region loop
  } // detail plot

  // NOTE Energy Loss
  booker.setCurrentFolder("MuonGEMHitsV/GEMHitsTask/EnergyLoss");

  for (const auto& station : gem->regions()[0]->stations()) {
    Int_t station_id = station->station();

    const char* eloss_name = TString::Format("eloss_muon_st%d", station_id);
    const char* eloss_title = TString::Format("SimHit Energy Loss (Muon only) : Station %d;Energy loss [eV];Entries", station_id);
    me_eloss_mu_[station_id] = booker.book1D(eloss_name, eloss_title, 60, 0.0, 6000.0);
  }  // station loop

  if (detail_plot_) {
    for (const auto& region : gem->regions()) {
      Int_t region_id = region->region();

      for (const auto& station : region->stations()) {
        Int_t station_id = station->station();

        const GEMSuperChamber* super_chamber = station->superChambers().front();
        for (const auto& chamber : super_chamber->chambers()) {
          Int_t layer_id = chamber->id().layer();
          ME3IdsKey key3(region_id, station_id, layer_id);

          me_detail_eloss_[key3] = bookHist1D(
              booker, key3, "eloss", "SimHit Energy Loss",
              60, 0.0, 6000.0, "Energy loss [eV]");

          me_detail_eloss_mu_[key3] = bookHist1D(
              booker, key3, "eloss_muon", "SimHit Energy Loss (Muon Only)",
              60, 0.0, 6000.0, "Energy loss [eV]");
        } // chamber loop
      } // station loop
    } // region loop
  } // detail plot

  // NOTE Occupancy
  booker.setCurrentFolder("MuonGEMHitsV/GEMHitsTask/Occupancy");

  for (const auto& region : gem->regions()) {
    Int_t region_id = region->region();

    me_occ_zr_[region_id] = bookZROccupancy(booker, region_id, "simhit", "SimHit");

    for (const auto& station : region->stations()) {
      Int_t station_id = station->station();
      ME2IdsKey key2(region_id, station_id);

       me_occ_det_[key2] = bookDetectorOccupancy(booker, key2, station, "simhit", "SimHit");

      const GEMSuperChamber* super_chamber = station->superChambers().front();
      for (const auto& chamber : super_chamber->chambers()) {
        Int_t layer_id = chamber->id().layer();
        ME3IdsKey key3(region_id, station_id, layer_id);

        // FIXME me_occ_xy_[key3] = bookXYOccupancy(booker, key3, "simhit", "SimHit");
      } // layer loop
    } // station loop
  } // region loop 

  return;
}


std::tuple<Double_t, Double_t> GEMSimHitValidation::getTOFRange(Int_t station_id) {
  UInt_t start_index = station_id == 1 ? 0 : 2;
  Double_t tof_min = tof_range_[start_index];
  Double_t tof_max = tof_range_[start_index + 1];
  return std::make_tuple(tof_min, tof_max);
}


void GEMSimHitValidation::analyze(const edm::Event& event, const edm::EventSetup& setup) {

  const GEMGeometry* gem = initGeometry(setup);

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
    Int_t roll_id = gemid.roll(); 

    ME2IdsKey key2(region_id, station_id);
    ME3IdsKey key3(region_id, station_id, layer_id);

    LocalPoint&& simhit_local_pos = simhit.localPosition();
    GlobalPoint&& simhit_global_pos = gem->idToDet(gemid)->surface().toGlobal(simhit_local_pos);

    Float_t simhit_g_r = simhit_global_pos.perp();
    // Float_t simhit_g_x = simhit_global_pos.x();
    // Float_t simhit_g_y = simhit_global_pos.y();
    Float_t simhit_g_abs_z = std::fabs(simhit_global_pos.z());

    Float_t energy_loss = kEnergyCF_ * simhit.energyLoss();
    Float_t tof = simhit.timeOfFlight();

    // NOTE Fill MonitorElement
    Int_t bin_x = getDetOccBinX(chamber_id, layer_id);

    me_occ_zr_[region_id]->Fill(simhit_g_abs_z, simhit_g_r);
    me_occ_det_[key2]->Fill(bin_x, roll_id);
    // FIXME me_occ_xy_[key3]->Fill(simhit_g_x, simhit_g_y);

    bool is_muon_simhit = isMuonSimHit(simhit);
    if (is_muon_simhit) {
      me_tof_mu_[station_id]->Fill(tof);
      me_eloss_mu_[station_id]->Fill(energy_loss);
    }

    if (detail_plot_) {
      me_detail_tof_[key3]->Fill(tof);
      me_detail_eloss_[key3]->Fill(energy_loss);

      if (is_muon_simhit) {
        me_detail_tof_mu_[key3]->Fill(tof);
        me_detail_eloss_mu_[key3]->Fill(energy_loss);
      }

    } // detail_plot
  } // simhit loop 
}
