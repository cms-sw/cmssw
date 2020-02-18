#include "Validation/MuonGEMHits/interface/MuonGEMBaseHarvestor.h"

#include "TEfficiency.h"


MuonGEMBaseHarvestor::MuonGEMBaseHarvestor(const edm::ParameterSet& pset,
                                           std::string log_category)
    : kLogCategory_(log_category) {
}


TProfile* MuonGEMBaseHarvestor::computeEfficiency(
    const TH1F& passed, const TH1F& total,
    const char* name, const char* title,
    Double_t confidence_level) {

  const TAxis* total_x = total.GetXaxis();

  TProfile* eff_profile = new TProfile(name, title, total_x->GetNbins(),
                                       total_x->GetXmin(), total_x->GetXmax());
  eff_profile->GetXaxis()->SetTitle(total_x->GetTitle());
  eff_profile->GetYaxis()->SetTitle("#epsilon");

  for (Int_t bin = 1; bin < total.GetXaxis()->GetNbins(); bin++) {
    Double_t num_passed = passed.GetBinContent(bin);
    Double_t num_total = total.GetBinContent(bin);

    if (num_passed > num_total) {
      edm::LogError(kLogCategory_) << "# of passed events > # of total events. "
                                   << "These numbers are not consistent." << std::endl;
      continue;
    }

    if (num_total < 1) {
      eff_profile->SetBinEntries(bin, 0);
      continue;
    }

    Double_t efficiency = num_passed / num_total;

    Double_t lower_bound = TEfficiency::ClopperPearson(num_total, num_passed, confidence_level, false);
    Double_t upper_bound = TEfficiency::ClopperPearson(num_total, num_passed, confidence_level, true);

    Double_t width = std::max(efficiency - lower_bound, upper_bound - efficiency);
    Double_t error = std::hypot(efficiency, width);

    eff_profile->SetBinContent(bin, efficiency);
    eff_profile->SetBinError(bin, error);
    eff_profile->SetBinEntries(bin, 1);
  }

  return eff_profile;
}


TH2F* MuonGEMBaseHarvestor::computeEfficiency(const TH2F& passed,
                                              const TH2F& total,
                                              const char* name,
                                              const char* title) {
  TEfficiency eff(passed, total);
  TH2F* eff_hist = dynamic_cast<TH2F*>(eff.CreateHistogram());
  eff_hist->SetName(name);
  eff_hist->SetTitle(title);

  const TAxis* total_x = total.GetXaxis();
  TAxis* eff_hist_x = eff_hist->GetXaxis();
  eff_hist_x->SetTitle(total_x->GetTitle());
  for (Int_t bin = 1; bin <= total.GetNbinsX(); bin++) {
    const char* label = total_x->GetBinLabel(bin);
    eff_hist_x->SetBinLabel(bin, label);
  }

  const TAxis* total_y = total.GetYaxis();
  TAxis* eff_hist_y = eff_hist->GetYaxis();
  eff_hist_y->SetTitle(total_y->GetTitle());
  for (Int_t bin = 1; bin <= total.GetNbinsY(); bin++) {
    const char* label = total_y->GetBinLabel(bin);
    eff_hist_y->SetBinLabel(bin, label);
  }

  return eff_hist;
}


void MuonGEMBaseHarvestor::bookEff1D(DQMStore::IBooker& booker,
                                     DQMStore::IGetter& getter,
                                     const TString& passed_path,
                                     const TString& total_path,
                                     const TString& folder,
                                     const TString& eff_name,
                                     const TString& eff_title) {
  TH1F* passed = getElement<TH1F>(getter, passed_path);
  if (passed == nullptr) {
    edm::LogError(kLogCategory_) << "failed to get " << passed_path << std::endl;
    return;
  }

  TH1F* total = getElement<TH1F>(getter, total_path);
  if (total == nullptr) {
    edm::LogError(kLogCategory_) << "failed to get " << total_path << std::endl;
    return;
  }

  TProfile* eff = computeEfficiency(*passed, *total, eff_name, eff_title);

  booker.setCurrentFolder(folder.Data());
  booker.bookProfile(eff_name, eff);
}


void MuonGEMBaseHarvestor::bookEff2D(DQMStore::IBooker& booker,
                                     DQMStore::IGetter& getter,
                                     const TString& passed_path,
                                     const TString& total_path,
                                     const TString& folder,
                                     const TString& eff_name,
                                     const TString& eff_title) {
  TH2F* passed = getElement<TH2F>(getter, passed_path);
  if (passed == nullptr) {
    edm::LogError(kLogCategory_) << "failed to get " << passed_path << std::endl;
    return;
  }

  TH2F* total = getElement<TH2F>(getter, total_path);
  if (total == nullptr) {
    edm::LogError(kLogCategory_) << "failed to get " << total_path << std::endl;
    return;
  }

  TH2F* eff = computeEfficiency(*passed, *total, eff_name, eff_title);
  booker.setCurrentFolder(folder.Data());
  booker.book2D(eff_name, eff);
}
