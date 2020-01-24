#include "Validation/MuonGEMRecHits/plugins/MuonGEMRecHitsHarvestor.h"
#include "Validation/MuonGEMHits/interface/GEMValidationUtils.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

MuonGEMRecHitsHarvestor::MuonGEMRecHitsHarvestor(const edm::ParameterSet& ps) : MuonGEMBaseHarvestor(ps) {
  folder_ = ps.getParameter<std::string>("folder");

  region_ids_ = ps.getUntrackedParameter<std::vector<Int_t> >("regionIds");
  station_ids_ = ps.getUntrackedParameter<std::vector<Int_t> >("stationIds");
  layer_ids_ = ps.getUntrackedParameter<std::vector<Int_t> >("layerIds");
}

MuonGEMRecHitsHarvestor::~MuonGEMRecHitsHarvestor() {}

void MuonGEMRecHitsHarvestor::dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) {
  const char* occ_folder = gSystem->ConcatFileName(folder_.c_str(), "Occupancy");
  const char* eff_folder = gSystem->ConcatFileName(folder_.c_str(), "Efficiency");
  booker.setCurrentFolder(eff_folder);

  for (const auto& region_id : region_ids_) {
    TString name_suffix_re = GEMUtils::getSuffixName(region_id);
    TString title_suffix_re = GEMUtils::getSuffixTitle(region_id);

    TString rechit_eta_path = gSystem->ConcatFileName(occ_folder, "matched_rechit_occ_eta" + name_suffix_re);

    TString simhit_eta_path = gSystem->ConcatFileName(occ_folder, "muon_simhit_occ_eta" + name_suffix_re);

    TString eff_eta_name = "eff_eta" + name_suffix_re;
    TString eff_eta_title = "Eta Efficiency :" + title_suffix_re;

    bookEff1D(booker, getter, rechit_eta_path, simhit_eta_path, eff_folder, eff_eta_name, eff_eta_title);

    for (const auto& station_id : station_ids_) {
      TString name_suffix_re_st = GEMUtils::getSuffixName(region_id, station_id);
      TString title_suffix_re_st = GEMUtils::getSuffixTitle(region_id, station_id);

      TString rechit_phi_path = gSystem->ConcatFileName(occ_folder, "matched_rechit_occ_phi" + name_suffix_re_st);

      TString simhit_phi_path = gSystem->ConcatFileName(occ_folder, "muon_simhit_occ_phi" + name_suffix_re_st);

      TString eff_phi_name = "eff_phi" + name_suffix_re_st;
      TString eff_phi_title = "Phi Efficiency :" + title_suffix_re_st;

      bookEff1D(booker, getter, rechit_phi_path, simhit_phi_path, eff_folder, eff_phi_name, eff_phi_title);

      // NOTE Detector Component Efficiency
      TString rechit_det_path = gSystem->ConcatFileName(occ_folder, "matched_rechit_occ_det" + name_suffix_re_st);

      TString simhit_det_path = gSystem->ConcatFileName(occ_folder, "muon_simhit_occ_det" + name_suffix_re_st);

      TString eff_det_name = "eff_det" + name_suffix_re_st;
      TString eff_det_title = "Detector Component Efficiency :" + title_suffix_re_st;
      bookEff2D(booker, getter, rechit_det_path, simhit_det_path, eff_folder, eff_det_name, eff_det_title);

    }  // Station Id END
  }    // Region Id END
}
