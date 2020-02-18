#ifndef Validation_MuonGEMHits_GEMBaseValidation_h
#define Validation_MuonGEMHits_GEMBaseValidation_h

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Validation/MuonGEMHits/interface/GEMValidationUtils.h"

#include "TMath.h"

using namespace dqm::impl; // MonitorElement


class GEMBaseValidation : public DQMEDAnalyzer {
 public:
  explicit GEMBaseValidation(const edm::ParameterSet&, std::string);
  ~GEMBaseValidation() override = 0;
  void analyze(const edm::Event& e, const edm::EventSetup&) override = 0;

 protected:
  const GEMGeometry* initGeometry(const edm::EventSetup&);
  Int_t getDetOccBinX(Int_t chamber_id, Int_t layer_id);
  Bool_t isMuonSimHit(const PSimHit&);

  MonitorElement* bookZROccupancy(DQMStore::IBooker& booker,
                                  Int_t region_id,
                                  const char* name_prfix,
                                  const char* title_prefix);

  template <typename T>
  MonitorElement* bookZROccupancy(DQMStore::IBooker& booker,
                                  const T& key,
                                  const char* name_prfix,
                                  const char* title_prefix);

  template <typename T>
  MonitorElement* bookXYOccupancy(DQMStore::IBooker& booker,
                                  const T& key,
                                  const char* name_prefix,
                                  const char* title_prefix);

  template <typename T>
  MonitorElement* bookPolarOccupancy(DQMStore::IBooker& booker,
                                     const T& key,
                                     const char* name_prefix,
                                     const char* title_prefix);

  template <typename T>
  MonitorElement* bookDetectorOccupancy(DQMStore::IBooker& booker,
                                        const T& key,
                                        const GEMStation* station,
                                        const char* name_prfix,
                                        const char* title_prefix);

  template <typename T>
  MonitorElement* bookHist1D(DQMStore::IBooker& booker,
                             const T& key,
                             const char* name,
                             const char* title,
                             Int_t nbinsx,
                             Double_t xlow,
                             Double_t xup,
                             const char* x_title = "",
                             const char* y_title = "Entries");

  template <typename T>
  MonitorElement* bookHist2D(DQMStore::IBooker& booker,
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
  std::vector<Int_t> zr_occ_num_bins_;
  std::vector<Double_t> zr_occ_range_;
  std::vector<Double_t> eta_range_;
  Bool_t detail_plot_;

  // NOTE Constants
  const Int_t kMuonPDGId_ = 13;
  const std::string kLogCategory_; // see member initializer list
};


template <typename T>
MonitorElement* GEMBaseValidation::bookZROccupancy(DQMStore::IBooker& booker,
                                                   const T& key,
                                                   const char* name_prefix,
                                                   const char* title_prefix) {
  if (std::tuple_size<T>::value < 2) {
    edm::LogError(kLogCategory_) << "Wrong T" << std::endl;
    return nullptr;
  }

  Int_t station_id = std::get<1>(key);

  const char* name_suffix = GEMUtils::getSuffixName(key).Data();
  const char* title_suffix = GEMUtils::getSuffixTitle(key).Data();

  TString name = TString::Format("%s_occ_zr%s", name_prefix, name_suffix);
  TString title = TString::Format("%s ZR Occupancy :%s;|Z| #[cm];R [cm]", title_prefix, title_suffix);

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
MonitorElement* GEMBaseValidation::bookXYOccupancy(DQMStore::IBooker& booker,
                                                   const T& key,
                                                   const char* name_prefix,
                                                   const char* title_prefix) {
  const char* name_suffix = GEMUtils::getSuffixName(key);
  const char* title_suffix = GEMUtils::getSuffixTitle(key);
  TString name = TString::Format("%s_occ_xy%s", name_prefix, name_suffix);
  TString title = TString::Format("%s XY Occupancy :%s;X [cm];Y [cm]", title_prefix, title_suffix);
  return booker.book2D(name, title, xy_occ_num_bins_, -360.0, 360.0, xy_occ_num_bins_, -360.0f, 360.0);
}


template <typename T>
MonitorElement* GEMBaseValidation::bookPolarOccupancy(DQMStore::IBooker& booker,
                                                      const T& key,
                                                      const char* name_prefix,
                                                      const char* title_prefix) {
  const char* name_suffix = GEMUtils::getSuffixName(key);
  const char* title_suffix = GEMUtils::getSuffixTitle(key);
  TString name = TString::Format("%s_occ_polar%s", name_prefix, name_suffix);
  TString title = TString::Format("%s Polar Occupancy :%s", title_prefix, title_suffix);
  // TODO # of bins
  // TODO the x-axis lies in the cnter of Ch1
  MonitorElement* me = booker.book2D(name, title, 108, -M_PI, M_PI, 108, 0.0, 360.0);
  return me;
}


template <typename T>
MonitorElement* GEMBaseValidation::bookDetectorOccupancy(DQMStore::IBooker& booker,
                                                         const T& key,
                                                         const GEMStation* station,
                                                         const char* name_prefix,
                                                         const char* title_prefix) {
  const char* name_suffix = GEMUtils::getSuffixName(key).Data();
  const char* title_suffix = GEMUtils::getSuffixTitle(key).Data();

  TString name = TString::Format("%s_occ_det%s", name_prefix, name_suffix);
  TString title = TString::Format("%s Occupancy for detector component :%s", title_prefix, title_suffix);

  std::vector<const GEMSuperChamber*> superchambers = station->superChambers();

  Int_t num_superchambers = superchambers.size();
  Int_t num_chambers = superchambers.front()->nChambers();

  Int_t nbinsx = num_superchambers * num_chambers;
  Int_t nbinsy = superchambers.front()->chambers().front()->nEtaPartitions();

  auto hist = new TH2F(name, title, nbinsx, 1 - 0.5, nbinsx + 0.5, nbinsy, 1 - 0.5, nbinsy + 0.5);
  hist->GetXaxis()->SetTitle("Chamber-Layer");
  hist->GetYaxis()->SetTitle("Eta Partition");

  TAxis* x_axis = hist->GetXaxis();
  for (Int_t chamber_id = 1; chamber_id <= num_superchambers; chamber_id++) {
    for (Int_t layer_id = 1; layer_id <= num_chambers; layer_id++) {
      Int_t bin = getDetOccBinX(chamber_id, layer_id);
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
MonitorElement* GEMBaseValidation::bookHist1D(DQMStore::IBooker& booker,
                                              const T& key,
                                              const char* name,
                                              const char* title,
                                              Int_t nbinsx,
                                              Double_t xlow,
                                              Double_t xup,
                                              const char* x_title,
                                              const char* y_title) {
  const char* name_suffix = GEMUtils::getSuffixName(key);
  const char* title_suffix = GEMUtils::getSuffixTitle(key);
  TString hist_name = TString::Format("%s%s", name, name_suffix);
  TString hist_title = TString::Format("%s :%s;%s;%s", title, title_suffix, x_title, y_title);
  return booker.book1D(hist_name, hist_title, nbinsx, xlow, xup);
}


template <typename T>
MonitorElement* GEMBaseValidation::bookHist2D(DQMStore::IBooker& booker,
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
  const char* name_suffix = GEMUtils::getSuffixName(key);
  const char* title_suffix = GEMUtils::getSuffixTitle(key);
  TString hist_name = TString::Format("%s%s", name, name_suffix);
  TString hist_title = TString::Format("%s :%s;%s;%s", title, title_suffix, x_title, y_title);
  return booker.book2D(hist_name, hist_title, nbinsx, xlow, xup, nbinsy, ylow, yup);
}


#endif // Validation_MuonGEMHits_GEMBaseValidation_h
