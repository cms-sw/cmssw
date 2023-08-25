#ifndef Validation_MuonGEMHits_INTERFACE_GEMValidationUtils_h
#define Validation_MuonGEMHits_INTERFACE_GEMValidationUtils_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "TString.h"
#include "TSystem.h"

#include <map>
#include <tuple>

class TH1F;
class TH2F;
class TProfile;

typedef std::tuple<Int_t, Int_t> ME2IdsKey;
typedef std::tuple<Int_t, Int_t, Int_t> ME3IdsKey;
typedef std::tuple<Int_t, Int_t, Int_t, Int_t> ME4IdsKey;
typedef std::tuple<Int_t, Int_t, Int_t, Int_t, Int_t>
    ME5IdsKey;  // 0: region, 1: station, 2: later, 3: module, 4: chamber or iEta

typedef std::map<Int_t, dqm::impl::MonitorElement*> MEMap1Ids;
typedef std::map<ME2IdsKey, dqm::impl::MonitorElement*> MEMap2Ids;
typedef std::map<ME3IdsKey, dqm::impl::MonitorElement*> MEMap3Ids;
typedef std::map<ME4IdsKey, dqm::impl::MonitorElement*> MEMap4Ids;
typedef std::map<ME5IdsKey, dqm::impl::MonitorElement*> MEMap5Ids;

namespace GEMUtils {
  TString getSuffixName(Int_t region_id);
  TString getSuffixName(Int_t region_id, Int_t station_id);
  TString getSuffixName(Int_t region_id, Int_t station_id, Int_t layer_id);
  TString getSuffixName(Int_t region_id, Int_t station_id, Int_t layer_id, Int_t eta_id);

  TString getSuffixName(const ME2IdsKey& key);
  TString getSuffixName(const ME3IdsKey& key);
  TString getSuffixName(const ME4IdsKey& key);

  TString getSuffixTitle(Int_t region_id);
  TString getSuffixTitle(Int_t region_id, Int_t station_id);
  TString getSuffixTitle(Int_t region_id, Int_t station_id, Int_t layer_id);
  TString getSuffixTitle(Int_t region_id, Int_t station_id, Int_t layer_id, Int_t eta_id);

  TString getSuffixTitle(const ME2IdsKey& key);
  TString getSuffixTitle(const ME3IdsKey& key);
  TString getSuffixTitle(const ME4IdsKey& key);

}  // namespace GEMUtils

#endif  // Validation_MuonGEMHits_GEMValidationUtils_h
