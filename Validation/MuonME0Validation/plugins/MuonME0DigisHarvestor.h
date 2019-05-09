#ifndef MuonME0DigisHarvestor_H
#define MuonME0DigisHarvestor_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"

#include "Validation/MuonME0Validation/interface/ME0DigisValidation.h"
#include <TEfficiency.h>
#include <TGraphAsymmErrors.h>
#include <TProfile.h>

class MuonME0DigisHarvestor : public DQMEDHarvester {
public:
  /// constructor
  explicit MuonME0DigisHarvestor(const edm::ParameterSet &);
  /// destructor
  ~MuonME0DigisHarvestor() override;

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;
  void ProcessBooking(DQMStore::IBooker &, DQMStore::IGetter &, std::string nameHist, TH1F *num, TH1F *den);
  void ProcessBookingBKG(
      DQMStore::IBooker &ibooker, DQMStore::IGetter &ig, std::string nameHist, TH1F *hist, TH1F *hist2);
  TProfile *ComputeEff(TH1F *num, TH1F *denum, std::string nameHist);
  TH1F *ComputeBKG(TH1F *hist1, TH1F *hist2, std::string nameHist);

private:
  std::string dbe_path_;
};
#endif
