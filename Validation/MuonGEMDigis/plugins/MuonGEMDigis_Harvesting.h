#ifndef MuonGEMDigis_Harvesting_H
#define MuonGEMDigis_Harvesting_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include <TEfficiency.h>
class MuonGEMDigis_Harvesting : public edm::EDAnalyzer
{
public:
  /// constructor
  explicit MuonGEMDigis_Harvesting(const edm::ParameterSet&);
  /// destructor
  ~MuonGEMDigis_Harvesting();

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);

  virtual void beginJob() ;

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void endJob() ;

  virtual void endRun(const edm::Run&, const edm::EventSetup&) ;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
	virtual TProfile* ComputeEff( TH1F* num, TH1F* denum);
	virtual void ProcessBooking( DQMStore* dbe_, const char* label, TString suffix, TH1F* track_hist, TH1F* sh_hist );

private:

  DQMStore* dbe_;
  

};
#endif
