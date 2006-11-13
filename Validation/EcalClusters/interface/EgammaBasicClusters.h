#ifndef Validation_EcalClusters_EgammaBasicClusters_h
#define Validation_EcalClusters_EgammaBasicClusters_h

/**\class EgammaBasicClusters

 Description: SVSuite Basic Cluster Validation

 Implementation:
     \\\author: Michael A. Balazs, Nov 2006
*/
//
// $Id: EgammaBasicClusters.h,v 1.1 2006/11/08 16:04:01 mabalazs Exp $
//
#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

class EgammaBasicClusters : public edm::EDAnalyzer
{
	public:
      explicit EgammaBasicClusters( const edm::ParameterSet& );
      ~EgammaBasicClusters();

      virtual void analyze( const edm::Event&, const edm::EventSetup& );
      virtual void beginJob(edm::EventSetup const&);
      virtual void endJob();

	private:
			std::string outputFile_;
			std::string CMSSW_Version_;

			bool verboseDBE_;
			DaqMonitorBEInterface* dbe_;

      edm::InputTag hybridBarrelBasicClusterCollection_;
      edm::InputTag islandBarrelBasicClusterCollection_;
      edm::InputTag islandEndcapBasicClusterCollection_;
 
      MonitorElement* hist_HybridEB_BC_Size_;
      MonitorElement* hist_IslandEB_BC_Size_;
      MonitorElement* hist_IslandEE_BC_Size_;

      double hist_min_Size_;
      double hist_max_Size_;
      int    hist_bins_Size_;

      MonitorElement* hist_HybridEB_BC_NumRecHits_;
      MonitorElement* hist_IslandEB_BC_NumRecHits_;
      MonitorElement* hist_IslandEE_BC_NumRecHits_;

      double hist_min_NumRecHits_;
      double hist_max_NumRecHits_;
      int    hist_bins_NumRecHits_;

      MonitorElement* hist_HybridEB_BC_ET_;
      MonitorElement* hist_IslandEB_BC_ET_;
      MonitorElement* hist_IslandEE_BC_ET_;

      double hist_min_ET_;
      double hist_max_ET_;
      int    hist_bins_ET_;

      MonitorElement* hist_HybridEB_BC_Eta_;
      MonitorElement* hist_IslandEB_BC_Eta_;
      MonitorElement* hist_IslandEE_BC_Eta_;

      double hist_min_Eta_;
      double hist_max_Eta_;
      int    hist_bins_Eta_;

      MonitorElement* hist_HybridEB_BC_Phi_;
      MonitorElement* hist_IslandEB_BC_Phi_;
      MonitorElement* hist_IslandEE_BC_Phi_;

      double hist_min_Phi_;
      double hist_max_Phi_;
      int    hist_bins_Phi_;
};
#endif
