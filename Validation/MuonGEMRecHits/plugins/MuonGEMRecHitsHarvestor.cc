#include "Validation/MuonGEMRecHits/plugins/MuonGEMRecHitsHarvestor.h"
#include "Validation/MuonGEMHits/interface/GEMValidationUtils.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

MuonGEMRecHitsHarvestor::MuonGEMRecHitsHarvestor(const edm::ParameterSet& pset)
    : MuonGEMBaseHarvestor(pset, "MuonGEMRecHitsHarvestor") {
  region_ids_ = pset.getUntrackedParameter<std::vector<Int_t> >("regionIds");
  station_ids_ = pset.getUntrackedParameter<std::vector<Int_t> >("stationIds");
  layer_ids_ = pset.getUntrackedParameter<std::vector<Int_t> >("layerIds");

  detail_plot_ = pset.getParameter<Bool_t>("detailPlot");
}

MuonGEMRecHitsHarvestor::~MuonGEMRecHitsHarvestor() {}

void MuonGEMRecHitsHarvestor::dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) {
  TString simhit_occ_folder = "MuonGEMHitsV/GEMHitsTask/Occupancy/";
  TString occ_folder = "MuonGEMRecHitsV/GEMRecHitsTask/Occupancy/";
  TString eff_folder = "MuonGEMRecHitsV/GEMRecHitsTask/Efficiency/";

  for (const auto& region_id : region_ids_) {
    for (const auto& station_id : station_ids_) {
      TString name_suf_re_st = GEMUtils::getSuffixName(region_id, station_id);
      TString title_suf_re_st = GEMUtils::getSuffixTitle(region_id, station_id);

      if (detail_plot_) {
        // NOTE Detector Component
        TString rechit_det_path = occ_folder + "matched_rechit_occ_det" + name_suf_re_st;
        TString simhit_det_path = simhit_occ_folder + "muon_simhit_occ_det" + name_suf_re_st;

        TString eff_det_name = "eff_det" + name_suf_re_st;
        TString eff_det_title = "Detector Component Efficiency :" + title_suf_re_st;

        bookEff2D(booker, getter, rechit_det_path, simhit_det_path, eff_folder, eff_det_name, eff_det_title);
      }
      for (const auto& layer_id : layer_ids_) {
        if (station_id != 0 and layer_id > 2)
          continue;
        TString name_suf_re_st_ly = GEMUtils::getSuffixName(region_id, station_id, layer_id);
        TString title_suf_re_st_ly = GEMUtils::getSuffixTitle(region_id, station_id, layer_id);

        // NOTE Eta
        TString rechit_eta_path = occ_folder + "matched_rechit_occ_eta" + name_suf_re_st_ly;
        TString simhit_eta_path = simhit_occ_folder + "muon_simhit_occ_eta" + name_suf_re_st_ly;

        TString eff_eta_name = "eff_eta" + name_suf_re_st_ly;
        TString eff_eta_title = "Eta Efficiency :" + title_suf_re_st_ly;

        bookEff1D(booker, getter, rechit_eta_path, simhit_eta_path, eff_folder, eff_eta_name, eff_eta_title);

        // NOTE Phi
        TString rechit_phi_path = occ_folder + "matched_rechit_occ_phi" + name_suf_re_st_ly;
        TString simhit_phi_path = simhit_occ_folder + "muon_simhit_occ_phi" + name_suf_re_st_ly;

        TString eff_phi_name = "eff_phi" + name_suf_re_st_ly;
        TString eff_phi_title = "Phi Efficiency :" + title_suf_re_st_ly;

        bookEff1D(booker, getter, rechit_phi_path, simhit_phi_path, eff_folder, eff_phi_name, eff_phi_title);
      }  // layer loop
    }    // station loop
  }      // region loop
}
