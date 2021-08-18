#include "Validation/MuonGEMHits/interface/GEMValidationUtils.h"

#include "TString.h"

TString GEMUtils::getSuffixName(Int_t region_id) { return TString::Format("_Re%+d", region_id); }

TString GEMUtils::getSuffixName(Int_t region_id, Int_t station_id) {
  return TString::Format("_GE%.2d-%c", station_id * 10 + 1, (region_id > 0 ? 'P' : 'M'));
}

TString GEMUtils::getSuffixName(Int_t region_id, Int_t station_id, Int_t layer_id) {
  return TString::Format("_GE%.2d-%c-L%d", station_id * 10 + 1, (region_id > 0 ? 'P' : 'M'), layer_id);
}

TString GEMUtils::getSuffixName(Int_t region_id, Int_t station_id, Int_t layer_id, Int_t eta_id) {
  return TString::Format("_GE%.2d-%c-L%d-E%d", station_id * 10 + 1, (region_id > 0 ? 'P' : 'M'), layer_id, eta_id);
}

TString GEMUtils::getSuffixName(const ME2IdsKey& key) {
  auto [region_id, station_id] = key;
  return getSuffixName(region_id, station_id);
}

TString GEMUtils::getSuffixName(const ME3IdsKey& key) {
  auto [region_id, station_id, layer_id] = key;
  return getSuffixName(region_id, station_id, layer_id);
}

TString GEMUtils::getSuffixName(const ME4IdsKey& key) {
  auto [region_id, station_id, layer_id, eta_id] = key;
  return getSuffixName(region_id, station_id, layer_id, eta_id);
}

TString GEMUtils::getSuffixTitle(Int_t region_id) { return TString::Format(" Region %+d", region_id); }

TString GEMUtils::getSuffixTitle(Int_t region_id, Int_t station_id) {
  return TString::Format(" GE%.2d-%c", station_id * 10 + 1, (region_id > 0 ? 'P' : 'M'));
}

TString GEMUtils::getSuffixTitle(Int_t region_id, Int_t station_id, Int_t layer_id) {
  return TString::Format(" GE%.2d-%c-L%d", station_id * 10 + 1, (region_id > 0 ? 'P' : 'M'), layer_id);
}

TString GEMUtils::getSuffixTitle(Int_t region_id, Int_t station_id, Int_t layer_id, Int_t eta_id) {
  return TString::Format(" GE%.2d-%c-L%d-E%d", station_id * 10 + 1, (region_id > 0 ? 'P' : 'M'), layer_id, eta_id);
}

TString GEMUtils::getSuffixTitle(const ME2IdsKey& key) {
  auto [region_id, station_id] = key;
  return getSuffixTitle(region_id, station_id);
}

TString GEMUtils::getSuffixTitle(const ME3IdsKey& key) {
  auto [region_id, station_id, layer_id] = key;
  return getSuffixTitle(region_id, station_id, layer_id);
}

TString GEMUtils::getSuffixTitle(const ME4IdsKey& key) {
  auto [region_id, station_id, layer_id, eta_id] = key;
  return getSuffixTitle(region_id, station_id, layer_id, eta_id);
}
