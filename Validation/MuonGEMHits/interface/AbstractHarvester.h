#ifndef AbstractHarvester_H
#define AbstractHarvester_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"


#include <TEfficiency.h>
#include <TGraphAsymmErrors.h>
#include <TProfile.h>
class AbstractHarvester : public edm::EDAnalyzer
{
public:
  /// constructor
	AbstractHarvester(){};
  explicit AbstractHarvester(const edm::ParameterSet&){};
  /// destructor
  virtual ~AbstractHarvester(){} ;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  virtual TProfile* ComputeEff( TH1F* num, TH1F* denum);
  virtual void ProcessBooking( std::string label_suffix, TH1F* track_hist, TH1F* sh_hist=nullptr ); 

protected:
	DQMStore* dbe_;
	std::string dbe_path_;
  std::string outputFile_;
	
};
#endif
