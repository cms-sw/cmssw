#ifndef MuonGEMHitsHarvestor_H
#define MuonGEMHitsHarvestor_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "DQMServices/Core/interface/MonitorElement.h"
//#include "Validation/MuonGEMHits/interface/GEMHitsValidation.h"
//#include "Validation/MuonGEMHits/interface/AbstractHarvester.h"
#include <TEfficiency.h>
#include <TGraphAsymmErrors.h>
#include <TProfile.h>

class MuonGEMHitsHarvestor : public DQMEDHarvester
{
public:
  /// constructor
  explicit MuonGEMHitsHarvestor(const edm::ParameterSet&);
  /// destructor
  virtual ~MuonGEMHitsHarvestor();

  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &);
  void ProcessBooking( DQMStore::IBooker& , DQMStore::IGetter&, std::string label_suffix, TH1F* track_hist, TH1F* sh_hist=nullptr );
  TProfile* ComputeEff(TH1F* num, TH1F* denum );
  

private:
  std::string dbe_path_,outputFile_;
};
#endif
