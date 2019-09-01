#ifndef MuonME0SegHarvestor_H
#define MuonME0SegHarvestor_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

#include <TEfficiency.h>
#include <TGraphAsymmErrors.h>
#include <TProfile.h>

#include "Validation/MuonME0Validation/interface/ME0RecHitsValidation.h"

class MuonME0SegHarvestor : public DQMEDHarvester {
public:
  /// constructor
  explicit MuonME0SegHarvestor(const edm::ParameterSet &);
  /// destructor
  ~MuonME0SegHarvestor() override;

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;
  void ProcessBooking(DQMStore::IBooker &, DQMStore::IGetter &, std::string nameHist, TH1F *num, TH1F *den);
  TProfile *ComputeEff(TH1F *num, TH1F *denum, std::string nameHist);

private:
  std::string dbe_path_;
};
#endif
