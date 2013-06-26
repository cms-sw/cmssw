#ifndef Validation_EcalClusters_EgammaBasicClusters_h
#define Validation_EcalClusters_EgammaBasicClusters_h

/**\class EgammaBasicClusters

 Description: SVSuite Basic Cluster Validation

 Implementation:
     \\\author: Michael A. Balazs, Nov 2006
*/
//
// $Id: EgammaBasicClusters.h,v 1.8 2009/12/14 22:24:32 wmtan Exp $
//
#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class EgammaBasicClusters : public edm::EDAnalyzer
{
	public:
      	explicit EgammaBasicClusters( const edm::ParameterSet& );
      	~EgammaBasicClusters();

      	virtual void analyze( const edm::Event&, const edm::EventSetup& );
      	virtual void beginJob();
      	virtual void endJob();

	private:
	std::string outputFile_;
	//std::string CMSSW_Version_;

	bool verboseDBE_;
	DQMStore* dbe_;

      	edm::InputTag barrelBasicClusterCollection_;
      	edm::InputTag endcapBasicClusterCollection_;
 
      	MonitorElement* hist_EB_BC_Size_;
      	MonitorElement* hist_EE_BC_Size_;

      	double hist_min_Size_;
      	double hist_max_Size_;
      	int    hist_bins_Size_;

      	MonitorElement* hist_EB_BC_NumRecHits_;
      	MonitorElement* hist_EE_BC_NumRecHits_;

      	double hist_min_NumRecHits_;
      	double hist_max_NumRecHits_;
      	int    hist_bins_NumRecHits_;

      	MonitorElement* hist_EB_BC_ET_;
      	MonitorElement* hist_EE_BC_ET_;

      	double hist_min_ET_;
      	double hist_max_ET_;
      	int    hist_bins_ET_;

      	MonitorElement* hist_EB_BC_Eta_;
      	MonitorElement* hist_EE_BC_Eta_;

      	double hist_min_Eta_;
      	double hist_max_Eta_;
      	int    hist_bins_Eta_;

      	MonitorElement* hist_EB_BC_Phi_;
      	MonitorElement* hist_EE_BC_Phi_;

      	double hist_min_Phi_;
      	double hist_max_Phi_;
      	int    hist_bins_Phi_;

      	double hist_min_R_;
      	double hist_max_R_;
      	int    hist_bins_R_;

	MonitorElement* hist_EB_BC_ET_vs_Eta_;
	MonitorElement* hist_EB_BC_ET_vs_Phi_;

	MonitorElement* hist_EE_BC_ET_vs_Eta_;
	MonitorElement* hist_EE_BC_ET_vs_Phi_;
	MonitorElement* hist_EE_BC_ET_vs_R_;


};
#endif
