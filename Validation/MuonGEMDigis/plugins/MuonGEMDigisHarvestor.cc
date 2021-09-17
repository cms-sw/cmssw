#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Validation/MuonGEMDigis/plugins/MuonGEMDigisHarvestor.h"

MuonGEMDigisHarvestor::MuonGEMDigisHarvestor(const edm::ParameterSet& pset)
    : MuonGEMBaseHarvestor(pset, "MuonGEMDigisHarvestor") {
  // to make it compatible to both full geometry and slice test
  region_ids_ = pset.getUntrackedParameter<std::vector<Int_t> >("regionIds");
  station_ids_ = pset.getUntrackedParameter<std::vector<Int_t> >("stationIds");
  layer_ids_ = pset.getUntrackedParameter<std::vector<Int_t> >("layerIds");

  detail_plot_ = pset.getParameter<Bool_t>("detailPlot");
}

MuonGEMDigisHarvestor::~MuonGEMDigisHarvestor() {}

void MuonGEMDigisHarvestor::dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) {
  TString simhit_occ_folder = "GEM/SimHits/";
  TString occ_folder = "GEM/Digis/";
  TString eff_folder = "GEM/Digis/";
  TString occ_folder_pad = "GEM/Pad/";
  TString eff_folder_pad = "GEM/Pad/";
  TString occ_folder_cluster = "GEM/PadCluster/";
  TString eff_folder_cluster = "GEM/PadCluster/";

  for (Int_t region_id : region_ids_) {
    for (Int_t station_id : station_ids_) {
      TString name_suffix_re_st = GEMUtils::getSuffixName(region_id, station_id);
      TString title_suffix_re_st = GEMUtils::getSuffixTitle(region_id, station_id);

      if (detail_plot_) {
        // NOTE Detector Component efficiency
        TString strip_det_name = "sim_matched_occ_det" + name_suffix_re_st;
        TString pad_det_name = "sim_matched_occ_det" + name_suffix_re_st;
        TString simhit_det_name = "sim_muon_occ_det" + name_suffix_re_st;
        TString strip_det_path = occ_folder + strip_det_name;
        TString pad_det_path = occ_folder_pad + pad_det_name;
        TString cluster_det_path = occ_folder_cluster + pad_det_name;
        TString simhit_det_path = simhit_occ_folder + simhit_det_name;
        TString eff_det_name = "eff_det" + name_suffix_re_st;
        TString eff_det_title = "Detector Component Efficiency (Muon Only) :" + title_suffix_re_st;

        bookEff2D(booker, getter, strip_det_path, simhit_det_path, eff_folder, eff_det_name, eff_det_title);
        bookEff2D(booker, getter, pad_det_path, simhit_det_path, eff_folder_pad, eff_det_name, eff_det_title);
        bookEff2D(booker, getter, cluster_det_path, simhit_det_path, eff_folder_cluster, eff_det_name, eff_det_title);
      }

      for (Int_t layer_id : layer_ids_) {
        if (station_id != 0 and layer_id > 2)
          continue;
        TString name_suffix_re_st_ly = GEMUtils::getSuffixName(region_id, station_id, layer_id);
        TString title_suffix_re_st_ly = GEMUtils::getSuffixTitle(region_id, station_id, layer_id);

        // NOTE eta efficiency
        TString strip_eta_name = "sim_matched_occ_eta" + name_suffix_re_st_ly;
        TString pad_eta_name = "sim_matched_occ_eta" + name_suffix_re_st_ly;
        TString simhit_eta_name = "sim_muon_occ_eta" + name_suffix_re_st_ly;
        TString strip_eta_path = occ_folder + strip_eta_name;
        TString pad_eta_path = occ_folder_pad + pad_eta_name;
        TString cluster_eta_path = occ_folder_cluster + pad_eta_name;
        TString simhit_eta_path = simhit_occ_folder + simhit_eta_name;
        TString eff_eta_name = "eff_eta" + name_suffix_re_st_ly;
        TString eff_eta_title = "Eta Efficiency (Muon Only) :" + title_suffix_re_st_ly;

        if (detail_plot_)
          bookEff1D(booker, getter, strip_eta_path, simhit_eta_path, eff_folder, eff_eta_name, eff_eta_title);
        bookEff1D(booker, getter, pad_eta_path, simhit_eta_path, eff_folder_pad, eff_eta_name, eff_eta_title);
        bookEff1D(booker, getter, cluster_eta_path, simhit_eta_path, eff_folder_cluster, eff_eta_name, eff_eta_title);

        // NOTE phi efficiency
        TString strip_phi_name = "sim_matched_occ_phi" + name_suffix_re_st_ly;
        TString pad_phi_name = "sim_matched_occ_phi" + name_suffix_re_st_ly;
        TString simhit_phi_name = "sim_muon_occ_phi" + name_suffix_re_st_ly;
        TString strip_phi_path = occ_folder + strip_phi_name;
        TString pad_phi_path = occ_folder_pad + pad_phi_name;
        TString cluster_phi_path = occ_folder_cluster + pad_phi_name;
        TString simhit_phi_path = simhit_occ_folder + simhit_phi_name;
        TString eff_phi_name = "eff_phi" + name_suffix_re_st_ly;
        TString eff_phi_title = "Phi Efficiency (Muon Only) :" + title_suffix_re_st_ly;

        if (detail_plot_)
          bookEff1D(booker, getter, strip_phi_path, simhit_phi_path, eff_folder, eff_phi_name, eff_phi_title);
        bookEff1D(booker, getter, pad_phi_path, simhit_phi_path, eff_folder_pad, eff_phi_name, eff_phi_title);
        bookEff1D(booker, getter, cluster_phi_path, simhit_phi_path, eff_folder_cluster, eff_phi_name, eff_phi_title);
      }  // layer loop
    }    // statino loop
  }      // region loop
}
