#ifndef Validation_MuonGEMDigis_MuonGEMDigisHarvestor_H
#define Validation_MuonGEMDigis_MuonGEMDigisHarvestor_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "Validation/MuonGEMDigis/plugins/GEMStripDigiValidation.h"
#include "Validation/MuonGEMDigis/plugins/GEMPadDigiValidation.h"
#include "Validation/MuonGEMDigis/plugins/GEMPadDigiClusterValidation.h"
#include "Validation/MuonGEMDigis/plugins/GEMCoPadDigiValidation.h"
#include <TEfficiency.h>
#include <TGraphAsymmErrors.h>
#include <TProfile.h>

class MuonGEMDigisHarvestor : public DQMEDHarvester {
public:
  /// constructor
  explicit MuonGEMDigisHarvestor(const edm::ParameterSet&);
  /// destructor
  ~MuonGEMDigisHarvestor() override;

  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;
  void ProcessBooking(
      DQMStore::IBooker&, DQMStore::IGetter&, const char* label, TString suffix, TH1F* track_hist, TH1F* sh_hist);
  TProfile* ComputeEff(TH1F* num, TH1F* denum);

private:
  std::string dbe_path_, dbe_hist_prefix_, compareable_dbe_path_, compareable_dbe_hist_prefix_, outputFile_;
};
#endif
