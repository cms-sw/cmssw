#ifndef Validation_EcalClusters_EgammaSuperClusters_h
#define Validation_EcalClusters_EgammaSuperClusters_h

/**\class EgammaSuperClusters

 Description: SVSuite Super Cluster Validation

 Implementation:
     \\\author: Michael A. Balazs, Nov 2006
*/
//
// $Id: EgammaSuperClusters.h,v 1.4 2006/11/20 18:58:30 mabalazs Exp $
//
#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

class EgammaSuperClusters : public edm::EDAnalyzer
{
	public:
      	explicit EgammaSuperClusters( const edm::ParameterSet& );
      	~EgammaSuperClusters();

      	virtual void analyze( const edm::Event&, const edm::EventSetup& );
      	virtual void beginJob(edm::EventSetup const&);
      	virtual void endJob();

	private:
	std::string outputFile_;
	std::string CMSSW_Version_;

	bool verboseDBE_;
	DaqMonitorBEInterface* dbe_;

      	edm::InputTag MCTruthCollection_;
      	edm::InputTag hybridBarrelSuperClusterCollection_;
      	edm::InputTag islandBarrelSuperClusterCollection_;
      	edm::InputTag islandEndcapSuperClusterCollection_;
      	edm::InputTag hybridBarrelClusterShapeAssociation_;
      	edm::InputTag islandBarrelClusterShapeAssociation_;
      	edm::InputTag islandEndcapClusterShapeAssociation_;

      	MonitorElement* hist_HybridEB_SC_Size_;
      	MonitorElement* hist_IslandEB_SC_Size_;
      	MonitorElement* hist_IslandEE_SC_Size_;

      	double hist_min_Size_;
      	double hist_max_Size_;
      	int    hist_bins_Size_;

      	MonitorElement* hist_HybridEB_SC_NumBC_;
      	MonitorElement* hist_IslandEB_SC_NumBC_;
      	MonitorElement* hist_IslandEE_SC_NumBC_;

      	double hist_min_NumBC_;
      	double hist_max_NumBC_;
      	int    hist_bins_NumBC_;

      	MonitorElement* hist_HybridEB_SC_ET_;
      	MonitorElement* hist_IslandEB_SC_ET_;
      	MonitorElement* hist_IslandEE_SC_ET_;

      	double hist_min_ET_;
      	double hist_max_ET_;
      	int    hist_bins_ET_;

      	MonitorElement* hist_HybridEB_SC_Eta_;
      	MonitorElement* hist_IslandEB_SC_Eta_;
      	MonitorElement* hist_IslandEE_SC_Eta_;

      	double hist_min_Eta_;
      	double hist_max_Eta_;
      	int    hist_bins_Eta_;

      	MonitorElement* hist_HybridEB_SC_Phi_;
      	MonitorElement* hist_IslandEB_SC_Phi_;
      	MonitorElement* hist_IslandEE_SC_Phi_;

      	double hist_min_Phi_;
      	double hist_max_Phi_;
      	int    hist_bins_Phi_;

      	MonitorElement* hist_HybridEB_SC_S1toS9_;
      	MonitorElement* hist_IslandEB_SC_S1toS9_;
      	MonitorElement* hist_IslandEE_SC_S1toS9_;

      	double hist_min_S1toS9_;
      	double hist_max_S1toS9_;
      	int    hist_bins_S1toS9_;

      	MonitorElement* hist_HybridEB_SC_S25toE_;
      	MonitorElement* hist_IslandEB_SC_S25toE_;
      	MonitorElement* hist_IslandEE_SC_S25toE_;

      	double hist_min_S25toE_;
      	double hist_max_S25toE_;
      	int    hist_bins_S25toE_;

      	MonitorElement* hist_HybridEB_SC_EToverTruth_;
      	MonitorElement* hist_IslandEB_SC_EToverTruth_;
      	MonitorElement* hist_IslandEE_SC_EToverTruth_;

      	double hist_min_EToverTruth_;
      	double hist_max_EToverTruth_;
      	int    hist_bins_EToverTruth_;

      	MonitorElement* hist_HybridEB_SC_deltaEta_;
      	MonitorElement* hist_IslandEB_SC_deltaEta_;
      	MonitorElement* hist_IslandEE_SC_deltaEta_;

      	double hist_min_deltaEta_;
      	double hist_max_deltaEta_;
      	int    hist_bins_deltaEta_;

      	float ecalEta(float EtaParticle , float Zvertex, float plane_Radius);
};
#endif
