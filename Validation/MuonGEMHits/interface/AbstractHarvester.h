#ifndef AbstractHarvester_H
#define AbstractHarvester_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


#include "DQMServices/Core/interface/DQMStore.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "Validation/MuonGEMHits/interface/GEMHitsValidation.h"
#include "Validation/MuonGEMHits/interface/GEMSimTrackMatch.h"
#include <TEfficiency.h>
class AbstractHarvester : public edm::EDAnalyzer
{
public:
  /// constructor
	AbstractHarvester(){};
  explicit AbstractHarvester(const edm::ParameterSet&){};
  /// destructor
  virtual ~AbstractHarvester(){} ;
/*
  virtual void beginRun(edm::Run const&, edm::EventSetup const&)=0;

  virtual void beginJob()=0 ;

  virtual void analyze(const edm::Event&, const edm::EventSetup&)=0;

  virtual void endJob()=0 ;

  virtual void endRun(const edm::Run&, const edm::EventSetup&)=0 ;
*/
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  virtual TProfile* ComputeEff( TH1F* num, TH1F* denum);
  virtual void ProcessBooking( std::string label_suffix, TH1F* track_hist, TH1F* sh_hist=nullptr ); 

protected:
	DQMStore* dbe_;
	std::string dbe_path_;
  std::string outputFile_;
	
};
#endif
