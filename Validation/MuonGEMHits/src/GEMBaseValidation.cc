#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include <memory>
using namespace std;
GEMBaseValidation::GEMBaseValidation(const edm::ParameterSet& ps) {
  zr_occ_num_bins_ = ps.getUntrackedParameter<std::vector<Int_t> >("ZROccNumBins");
  zr_occ_range_ = ps.getUntrackedParameter<std::vector<Double_t> >("ZROccRange");
  xy_occ_num_bins_ = ps.getUntrackedParameter<Int_t>("XYOccNumBins", 360);
  // TODO depends on the station.. for detail plots..
  eta_range_ = ps.getUntrackedParameter<std::vector<Double_t> >("EtaOccRange");

  folder_ = ps.getParameter<std::string>("folder");
  log_category_ = ps.getParameter<std::string>("logCategory");
  detail_plot_ = ps.getParameter<Bool_t>("detailPlot");

  const auto& pset = ps.getParameterSet("gemSimHit");
  inputTokenSH_ = consumes<edm::PSimHitContainer>(pset.getParameter<edm::InputTag>("inputTag"));
}

GEMBaseValidation::~GEMBaseValidation() {}

const GEMGeometry* GEMBaseValidation::initGeometry(edm::EventSetup const& event_setup) {
  edm::ESHandle<GEMGeometry> geom_handle;
  event_setup.get<MuonGeometryRecord>().get(geom_handle);
  const GEMGeometry* gem = &*geom_handle;
  return gem;
}

Int_t GEMBaseValidation::getDetOccBinX(Int_t chamber_id, Int_t layer_id) { return 2 * chamber_id + layer_id - 2; }

MonitorElement* GEMBaseValidation::bookZROccupancy(DQMStore::IBooker& ibooker,
                                                   Int_t region_id,
                                                   const char* name_prefix,
                                                   const char* title_prefix) {
  const char* name_suffix = GEMUtils::getSuffixName(region_id).Data();
  const char* title_suffix = GEMUtils::getSuffixTitle(region_id).Data();

  TString name = TString::Format("%s_occ_zr%s", name_prefix, name_suffix);
  TString title = TString::Format("%s ZR Occupancy :%s;|Z| [cm];R [cm]", title_prefix, title_suffix);

  Double_t station1_xmin = zr_occ_range_[0];
  Double_t station1_xmax = zr_occ_range_[1];
  Double_t station2_xmin = zr_occ_range_[4];
  Double_t station2_xmax = zr_occ_range_[5];

  std::vector<Double_t> xbins_vector;
  for (Double_t i = station1_xmin - 1; i < station2_xmax + 1; i += 0.25) {
    if (i > station1_xmax + 1 and i < station2_xmin - 1)
      continue;
    xbins_vector.push_back(i);
  }

  Int_t nbinsx = xbins_vector.size() - 1;

  Int_t nbinsy = zr_occ_num_bins_[2];
  Double_t ylow = std::min(zr_occ_range_[2], zr_occ_range_[6]);
  Double_t yup = std::max(zr_occ_range_[3], zr_occ_range_[7]);

  auto hist = new TH2F(name, title, nbinsx, &xbins_vector[0], nbinsy, ylow, yup);
  return ibooker.book2D(name, hist);
}

Bool_t GEMBaseValidation::isMuonSimHit(const PSimHit& simhit) { return std::abs(simhit.particleType()) == kMuonPDGId_; }
