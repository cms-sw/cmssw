#ifndef Validation_MuonGEMHits_GEMBaseValidation_h
#define Validation_MuonGEMHits_GEMBaseValidation_h

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Validation/MuonGEMHits/interface/GEMValidationUtils.h"

#include "TMath.h"
#include "TDatabasePDG.h"

class GEMBaseValidation : public DQMEDAnalyzer {
public:
  explicit GEMBaseValidation(const edm::ParameterSet&, std::string);
  ~GEMBaseValidation() override = 0;
  void analyze(const edm::Event& e, const edm::EventSetup&) override = 0;

protected:
  Int_t getDetOccBinX(Int_t num_layers, Int_t chamber_id, Int_t layer_id);
  Bool_t isMuonSimHit(const PSimHit&);
  Float_t toDegree(Float_t radian);
  Int_t getPidIdx(Int_t pid);

  dqm::impl::MonitorElement* bookZROccupancy(DQMStore::IBooker& booker,
                                             Int_t region_id,
                                             const char* name_prfix,
                                             const char* title_prefix);

  template <typename T>
  dqm::impl::MonitorElement* bookZROccupancy(DQMStore::IBooker& booker,
                                             const T& key,
                                             const char* name_prfix,
                                             const char* title_prefix);

  template <typename T>
  dqm::impl::MonitorElement* bookXYOccupancy(DQMStore::IBooker& booker,
                                             const T& key,
                                             const char* name_prefix,
                                             const char* title_prefix);

  template <typename T>
  dqm::impl::MonitorElement* bookPolarOccupancy(DQMStore::IBooker& booker,
                                                const T& key,
                                                const char* name_prefix,
                                                const char* title_prefix);

  template <typename T>
  dqm::impl::MonitorElement* bookDetectorOccupancy(DQMStore::IBooker& booker,
                                                   const T& key,
                                                   const GEMStation* station,
                                                   const char* name_prfix,
                                                   const char* title_prefix);

  template <typename T>
  dqm::impl::MonitorElement* bookPIDHist(DQMStore::IBooker& booker, const T& key, const char* name, const char* title);

  template <typename T>
  dqm::impl::MonitorElement* bookPIDHist(
      DQMStore::IBooker& booker, const T& key, Int_t ieta, const char* name, const char* title);

  template <typename T>
  dqm::impl::MonitorElement* bookHist1D(DQMStore::IBooker& booker,
                                        const T& key,
                                        const char* name,
                                        const char* title,
                                        Int_t nbinsx,
                                        Double_t xlow,
                                        Double_t xup,
                                        const char* x_title = "",
                                        const char* y_title = "Entries");

  template <typename T>
  dqm::impl::MonitorElement* bookHist2D(DQMStore::IBooker& booker,
                                        const T& key,
                                        const char* name,
                                        const char* title,
                                        Int_t nbinsx,
                                        Double_t xlow,
                                        Double_t xup,
                                        Int_t nbinsy,
                                        Double_t ylow,
                                        Double_t yup,
                                        const char* x_title = "",
                                        const char* y_title = "");

  // NOTE Parameters
  Int_t xy_occ_num_bins_;
  std::vector<Int_t> pid_list_;
  std::vector<Int_t> zr_occ_num_bins_;
  std::vector<Double_t> zr_occ_range_;
  std::vector<Double_t> eta_range_;
  Bool_t detail_plot_;

  // NOTE Constants
  const Int_t kMuonPDGId_ = 13;
  const std::string kLogCategory_;  // see member initializer list
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomToken_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomTokenBeginRun_;
};

template <typename T>
dqm::impl::MonitorElement* GEMBaseValidation::bookZROccupancy(DQMStore::IBooker& booker,
                                                              const T& key,
                                                              const char* name_prefix,
                                                              const char* title_prefix) {
  if (std::tuple_size<T>::value < 2) {
    edm::LogError(kLogCategory_) << "Wrong T" << std::endl;
    return nullptr;
  }

  Int_t station_id = std::get<1>(key);

  auto name_suffix = GEMUtils::getSuffixName(key);
  auto title_suffix = GEMUtils::getSuffixTitle(key);

  TString name = TString::Format("%s_occ_zr%s", name_prefix, name_suffix.Data());
  TString title = TString::Format("%s ZR Occupancy :%s;|Z| #[cm];R [cm]", title_prefix, title_suffix.Data());

  // NOTE currently, only GE11 and GE21 are considered.
  // Look Validation/MuonGEMHits/python/MuonGEMCommonParameters_cfi.py
  UInt_t nbins_start = 2 * (station_id - 1);
  Int_t nbinsx = zr_occ_num_bins_[nbins_start];
  Int_t nbinsy = zr_occ_num_bins_[nbins_start + 1];

  // st1 xmin xmax, ymin, ymax | st2 xmin, xmax, ymin ymax
  UInt_t range_start = 4 * (station_id - 1);
  // absolute z axis
  Double_t xlow = zr_occ_range_[range_start];
  Double_t xup = zr_occ_range_[range_start + 1];
  // R axis
  Double_t ylow = zr_occ_range_[range_start + 2];
  Double_t yup = zr_occ_range_[range_start + 3];

  return booker.book2D(name, title, nbinsx, xlow, xup, nbinsy, ylow, yup);
}

template <typename T>
dqm::impl::MonitorElement* GEMBaseValidation::bookXYOccupancy(DQMStore::IBooker& booker,
                                                              const T& key,
                                                              const char* name_prefix,
                                                              const char* title_prefix) {
  auto name_suffix = GEMUtils::getSuffixName(key);
  auto title_suffix = GEMUtils::getSuffixTitle(key);
  TString name = TString::Format("%s_occ_xy%s", name_prefix, name_suffix.Data());
  TString title = TString::Format("%s XY Occupancy :%s;X [cm];Y [cm]", title_prefix, title_suffix.Data());
  return booker.book2D(name, title, xy_occ_num_bins_, -360.0, 360.0, xy_occ_num_bins_, -360.0f, 360.0);
}

template <typename T>
dqm::impl::MonitorElement* GEMBaseValidation::bookPolarOccupancy(DQMStore::IBooker& booker,
                                                                 const T& key,
                                                                 const char* name_prefix,
                                                                 const char* title_prefix) {
  auto name_suffix = GEMUtils::getSuffixName(key);
  auto title_suffix = GEMUtils::getSuffixTitle(key);
  TString name = TString::Format("%s_occ_polar%s", name_prefix, name_suffix.Data());
  TString title = TString::Format("%s Polar Occupancy :%s", title_prefix, title_suffix.Data());
  // TODO # of bins
  // TODO the x-axis lies in the cnter of Ch1
  dqm::impl::MonitorElement* me = booker.book2D(name, title, 108, -M_PI, M_PI, 108, 0.0, 360.0);
  return me;
}

template <typename T>
dqm::impl::MonitorElement* GEMBaseValidation::bookDetectorOccupancy(DQMStore::IBooker& booker,
                                                                    const T& key,
                                                                    const GEMStation* station,
                                                                    const char* name_prefix,
                                                                    const char* title_prefix) {
  auto name_suffix = GEMUtils::getSuffixName(key);
  auto title_suffix = GEMUtils::getSuffixTitle(key);

  TString name = TString::Format("%s_occ_det%s", name_prefix, name_suffix.Data());
  TString title = TString::Format("%s Occupancy for detector component :%s", title_prefix, title_suffix.Data());

  std::vector<const GEMSuperChamber*> superchambers = station->superChambers();

  Int_t num_superchambers = superchambers.size();
  Int_t num_chambers = 0;
  Int_t nbinsy = 0;
  if (num_superchambers > 0) {
    num_chambers = superchambers.front()->nChambers();
    if (num_chambers > 0)
      nbinsy = superchambers.front()->chambers().front()->nEtaPartitions();
  }
  Int_t nbinsx = num_superchambers * num_chambers;

  if (nbinsx <= 0)
    nbinsx = 20;  // Ensure histogram is not zero size
  if (nbinsy <= 0)
    nbinsy = 20;
  auto hist = new TH2F(name, title, nbinsx, 1 - 0.5, nbinsx + 0.5, nbinsy, 1 - 0.5, nbinsy + 0.5);
  hist->GetXaxis()->SetTitle("Chamber-Layer");
  hist->GetYaxis()->SetTitle("Eta Partition");

  TAxis* x_axis = hist->GetXaxis();
  for (Int_t chamber_id = 1; chamber_id <= num_superchambers; chamber_id++) {
    for (Int_t layer_id = 1; layer_id <= num_chambers; layer_id++) {
      Int_t bin = getDetOccBinX(num_chambers, chamber_id, layer_id);
      TString label = TString::Format("C%dL%d", chamber_id, layer_id);
      x_axis->SetBinLabel(bin, label);
    }
  }

  TAxis* y_axis = hist->GetYaxis();
  for (Int_t bin = 1; bin <= nbinsy; bin++) {
    y_axis->SetBinLabel(bin, TString::Itoa(bin, 10));
  }

  return booker.book2D(name, hist);
}

template <typename T>
dqm::impl::MonitorElement* GEMBaseValidation::bookPIDHist(DQMStore::IBooker& booker,
                                                          const T& key,
                                                          const char* name,
                                                          const char* title) {
  auto name_suffix = GEMUtils::getSuffixName(key);
  auto title_suffix = GEMUtils::getSuffixTitle(key);
  TString x_title = "Particle Type";
  TString y_title = "Particles";
  TString hist_name = TString::Format("%s%s", name, name_suffix.Data());
  TString hist_title = TString::Format("%s :%s;%s;%s", title, title_suffix.Data(), x_title.Data(), y_title.Data());
  Int_t nbinsx = pid_list_.size();
  auto hist = booker.book1D(hist_name, hist_title, nbinsx + 1, 0, nbinsx + 1);
  TDatabasePDG* pdgDB = TDatabasePDG::Instance();
  for (Int_t idx = 0; idx < nbinsx; idx++) {
    Int_t bin = idx + 1;
    auto particle_name = pdgDB->GetParticle(pid_list_[idx])->GetName();
    hist->setBinLabel(bin, particle_name);
  }
  hist->setBinLabel(nbinsx + 1, "others");
  return hist;
}

template <typename T>
dqm::impl::MonitorElement* GEMBaseValidation::bookPIDHist(
    DQMStore::IBooker& booker, const T& key, Int_t ieta, const char* name, const char* title) {
  auto name_suffix = GEMUtils::getSuffixName(key);
  auto title_suffix = GEMUtils::getSuffixTitle(key);
  TString x_title = "Particle Type";
  TString y_title = "Particles";
  TString hist_name = TString::Format("%s%s-E%d", name, name_suffix.Data(), ieta);
  TString hist_title =
      TString::Format("%s :%s-E%d;%s;%s", title, title_suffix.Data(), ieta, x_title.Data(), y_title.Data());
  Int_t nbinsx = pid_list_.size();
  auto hist = booker.book1D(hist_name, hist_title, nbinsx + 1, 0, nbinsx + 1);
  TDatabasePDG* pdgDB = TDatabasePDG::Instance();
  for (Int_t idx = 0; idx < nbinsx; idx++) {
    Int_t bin = idx + 1;
    auto particle_name = pdgDB->GetParticle(pid_list_[idx])->GetName();
    hist->setBinLabel(bin, particle_name);
  }
  hist->setBinLabel(nbinsx + 1, "others");
  return hist;
}

template <typename T>
dqm::impl::MonitorElement* GEMBaseValidation::bookHist1D(DQMStore::IBooker& booker,
                                                         const T& key,
                                                         const char* name,
                                                         const char* title,
                                                         Int_t nbinsx,
                                                         Double_t xlow,
                                                         Double_t xup,
                                                         const char* x_title,
                                                         const char* y_title) {
  auto name_suffix = GEMUtils::getSuffixName(key);
  auto title_suffix = GEMUtils::getSuffixTitle(key);
  TString hist_name = TString::Format("%s%s", name, name_suffix.Data());
  TString hist_title = TString::Format("%s :%s;%s;%s", title, title_suffix.Data(), x_title, y_title);
  return booker.book1D(hist_name, hist_title, nbinsx, xlow, xup);
}

template <typename T>
dqm::impl::MonitorElement* GEMBaseValidation::bookHist2D(DQMStore::IBooker& booker,
                                                         const T& key,
                                                         const char* name,
                                                         const char* title,
                                                         Int_t nbinsx,
                                                         Double_t xlow,
                                                         Double_t xup,
                                                         Int_t nbinsy,
                                                         Double_t ylow,
                                                         Double_t yup,
                                                         const char* x_title,
                                                         const char* y_title) {
  auto name_suffix = GEMUtils::getSuffixName(key);
  auto title_suffix = GEMUtils::getSuffixTitle(key);
  TString hist_name = TString::Format("%s%s", name, name_suffix.Data());
  TString hist_title = TString::Format("%s :%s;%s;%s", title, title_suffix.Data(), x_title, y_title);
  return booker.book2D(hist_name, hist_title, nbinsx, xlow, xup, nbinsy, ylow, yup);
}

#endif  // Validation_MuonGEMHits_GEMBaseValidation_h
