#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Validation/MuonGEMDigis/plugins/MuonGEMDigisHarvestor.h"
#include <string>
#include <vector>

MuonGEMDigisHarvestor::MuonGEMDigisHarvestor(const edm::ParameterSet& ps) : MuonGEMBaseHarvestor(ps) {
  strip_folder_ = ps.getUntrackedParameter<std::string>("stripFolder");
  pad_folder_ = ps.getUntrackedParameter<std::string>("padFolder");
  copad_folder_ = ps.getUntrackedParameter<std::string>("copadFolder");

  region_ids_ = ps.getUntrackedParameter<std::vector<Int_t> >("regionIds");
  station_ids_ = ps.getUntrackedParameter<std::vector<Int_t> >("stationIds");
  layer_ids_ = ps.getUntrackedParameter<std::vector<Int_t> >("layerIds");
}

MuonGEMDigisHarvestor::~MuonGEMDigisHarvestor() {}

void MuonGEMDigisHarvestor::dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) {
  std::cout << "MuonGEMDigisHarvestor::dqmEndJob begin" << std::endl;

  const char* occ_folder = gSystem->ConcatFileName(strip_folder_.c_str(), "Occupancy");
  const char* eff_folder = gSystem->ConcatFileName(strip_folder_.c_str(), "Efficiency");
  booker.setCurrentFolder(eff_folder);

  for (Int_t region_id : region_ids_) {
    TString name_suffix_re = GEMUtils::getSuffixName(region_id);
    TString title_suffix_re = GEMUtils::getSuffixTitle(region_id);

    // NOTE eta efficiency
    TString strip_eta_path = gSystem->ConcatFileName(occ_folder, "matched_strip_occ_eta" + name_suffix_re);

    TString simhit_eta_path = gSystem->ConcatFileName(occ_folder, "muon_simhit_occ_eta" + name_suffix_re);

    TString eff_eta_name = "eff_eta" + name_suffix_re;
    TString eff_eta_title = "Eta Efficiency (Muon Only) :" + title_suffix_re;

    bookEff1D(booker, getter, strip_eta_path, simhit_eta_path, eff_folder, eff_eta_name, eff_eta_title);

    for (Int_t station_id : station_ids_) {
      TString name_suffix_re_st = GEMUtils::getSuffixName(region_id, station_id);
      TString title_suffix_re_st = GEMUtils::getSuffixTitle(region_id, station_id);

      // NOTE phi efficiency
      TString strip_phi_path = gSystem->ConcatFileName(occ_folder, "matched_strip_occ_phi" + name_suffix_re_st);

      TString simhit_phi_path = gSystem->ConcatFileName(occ_folder, "muon_simhit_occ_phi" + name_suffix_re_st);

      TString eff_phi_name = "eff_phi" + name_suffix_re_st;
      TString eff_phi_title = "Phi Efficiency (Muon Only) :" + title_suffix_re;

      bookEff1D(booker, getter, strip_phi_path, simhit_phi_path, eff_folder, eff_phi_name, eff_phi_title);

      // NOTE Detector Component efficiency
      TString strip_det_path = gSystem->ConcatFileName(occ_folder, "matched_strip_occ_det" + name_suffix_re_st);

      TString simhit_det_path = gSystem->ConcatFileName(occ_folder, "muon_simhit_occ_det" + name_suffix_re_st);

      TString eff_det_name = "eff_det" + name_suffix_re_st;
      TString eff_det_title = "Detector Component Efficiency (Muon Only) :" + title_suffix_re_st;

      bookEff2D(booker, getter, strip_det_path, simhit_det_path, eff_folder, eff_det_name, eff_det_title);

    }  // end loop over station ids
  }    // end loop over region ids

  std::cout << "MuonGEMDigisHarvestor::dqmEndJob end" << std::endl;
}
