#ifndef MuonGEMHitsHarvestor_H
#define MuonGEMHitsHarvestor_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "Validation/MuonGEMHits/interface/GEMHitsValidation.h"
//#include "Validation/MuonGEMHits/interface/AbstractHarvester.h"

class MuonGEMHitsHarvestor : public DQMEDHarvester
{
public:
  /// constructor
  explicit MuonGEMHitsHarvestor(const edm::ParameterSet&);
  /// destructor
  virtual ~MuonGEMHitsHarvestor();

  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &);

private:
  std::string dbe_path_,outputFile_;
};
#endif
