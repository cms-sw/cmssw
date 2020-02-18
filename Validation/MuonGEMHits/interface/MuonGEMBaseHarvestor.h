#ifndef Validation_MuonGEMHits_MuonGEMBaseHarvestor_h
#define Validation_MuonGEMHits_MuonGEMBaseHarvestor_h

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "Validation/MuonGEMHits/interface/GEMValidationUtils.h"

#include "TSystem.h"
#include "TString.h"


class MuonGEMBaseHarvestor : public DQMEDHarvester {
 public:
  explicit MuonGEMBaseHarvestor(const edm::ParameterSet&, std::string);

 protected:
  // FIXME typename
  template <typename T>
  T* getElement(DQMStore::IGetter& getter, const TString& path);

  // 0.683 means 1 standard deviation of the normal distribution.
  TProfile* computeEfficiency(const TH1F& passed, const TH1F& total,
                              const char* name, const char* title,
                              Double_t confidence_level = 0.683);

  TH2F* computeEfficiency(const TH2F& passed, const TH2F& total,
                          const char* name, const char* title);

  void bookEff1D(DQMStore::IBooker& ibooker,
                 DQMStore::IGetter& getter,
                 const TString& passed_path,
                 const TString& total_path,
                 const TString& folder,
                 const TString& eff_name,
                 const TString& eff_title = "Efficiency");

  void bookEff2D(DQMStore::IBooker& ibooker,
                 DQMStore::IGetter& getter,
                 const TString& passed_path,
                 const TString& total_path,
                 const TString& folder,
                 const TString& eff_name,
                 const TString& eff_title = "Efficiency");


  // Constnats
  const std::string kLogCategory_; // see member initializer list
};

#endif // Validation_MuonGEMHits_MuonGEMBaseHarvestor_h


template <typename T>
T* MuonGEMBaseHarvestor::getElement(DQMStore::IGetter& getter,
                                    const TString& path) {
  std::string folder = gSystem->DirName(path);
  std::string name = gSystem->BaseName(path);

  getter.setCurrentFolder(folder);
  std::vector<std::string> mes = getter.getMEs();

  if (std::find(mes.begin(), mes.end(), name) == mes.end()) {
    edm::LogInfo(kLogCategory_) << "doesn't contain " << path << std::endl;
    return nullptr;
  }

  T* hist = nullptr;
  if (auto tmp_me = getter.get(path.Data())) {

    if (auto tmp_hist = dynamic_cast<T*>(tmp_me->getRootObject()->Clone())) {
      hist = tmp_hist;
      hist->Sumw2();
    } else {
      edm::LogError(kLogCategory_) << "failed to getRootObject " << path << std::endl;
    }

  } else {
    edm::LogError(kLogCategory_) << "failed to get " << path << std::endl;
  }

  return hist;
}
