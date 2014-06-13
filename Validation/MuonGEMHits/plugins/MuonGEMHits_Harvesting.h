#ifndef MuonGEMHits_Harvesting_H
#define MuonGEMHits_Harvesting_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


#include "DQMServices/Core/interface/DQMStore.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "Validation/MuonGEMHits/interface/GEMHitsValidation.h"
#include "Validation/MuonGEMHits/interface/GEMSimTrackMatch.h"
#include "Validation/MuonGEMHits/interface/AbstractHarvester.h"
class MuonGEMHits_Harvesting : public AbstractHarvester
{
public:
  /// constructor
  explicit MuonGEMHits_Harvesting(const edm::ParameterSet&);
  /// destructor
  ~MuonGEMHits_Harvesting();

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);

  virtual void beginJob() ;

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void endJob() ;

  virtual void endRun(const edm::Run&, const edm::EventSetup&) ;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
	//DQMStore* dbe_;
};
#endif
