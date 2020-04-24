#ifndef MuonGEMDigisHarvestor_H
#define MuonGEMDigisHarvestor_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "Validation/MuonGEMRecHits/interface/GEMRecHitsValidation.h"
#include <TEfficiency.h>
#include <TGraphAsymmErrors.h>
#include <TProfile.h>

class MuonGEMRecHitsHarvestor : public DQMEDHarvester
{
public:
  /// constructor
  explicit MuonGEMRecHitsHarvestor(const edm::ParameterSet&);
  /// destructor
  virtual ~MuonGEMRecHitsHarvestor();

  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &);
  void ProcessBooking( DQMStore::IBooker& , DQMStore::IGetter&, const char* label, TString suffix, TH1F* track_hist, TH1F* sh_hist );
  TProfile* ComputeEff(TH1F* num, TH1F* denum );

private:
  std::string dbe_path_,outputFile_;
};
#endif
