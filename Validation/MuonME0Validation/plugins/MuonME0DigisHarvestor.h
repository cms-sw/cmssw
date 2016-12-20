#ifndef MuonME0DigisHarvestor_H
#define MuonME0DigisHarvestor_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

#include "Validation/MuonME0Validation/interface/ME0DigisValidation.h"
#include <TEfficiency.h>
#include <TGraphAsymmErrors.h>
#include <TProfile.h>

class MuonME0DigisHarvestor : public DQMEDHarvester
{
public:
  /// constructor
  explicit MuonME0DigisHarvestor(const edm::ParameterSet&);
  /// destructor
  virtual MuonME0DigisHarvestor();

  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &);
  void ProcessBooking( DQMStore::IBooker& , DQMStore::IGetter&, const char* label, TString suffix, TH1F* num, TH1F* den );
  TProfile* ComputeEff(TH1F* num, TH1F* denum );

private:
  std::string dbe_path_,outputFile_;
};
#endif
