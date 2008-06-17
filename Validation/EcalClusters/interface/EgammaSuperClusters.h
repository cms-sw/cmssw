#ifndef Validation_EcalClusters_EgammaSuperClusters_h
#define Validation_EcalClusters_EgammaSuperClusters_h

/**\class EgammaSuperClusters

 Description: SVSuite Super Cluster Validation

 Implementation:
     \\\author: Michael A. Balazs, Nov 2006
*/
//
// $Id: EgammaSuperClusters.h,v 1.6 2008/02/29 20:48:23 ksmith Exp $
//
#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

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
	DQMStore* dbe_;

      	edm::InputTag MCTruthCollection_;
      	edm::InputTag barrelSuperClusterCollection_;
      	edm::InputTag endcapSuperClusterCollection_;
      	edm::InputTag barrelRecHitCollection_;
      	edm::InputTag endcapRecHitCollection_;

      	MonitorElement* hist_EB_SC_Size_;
      	MonitorElement* hist_EE_SC_Size_;

      	double hist_min_Size_;
      	double hist_max_Size_;
      	int    hist_bins_Size_;

      	MonitorElement* hist_EB_SC_NumBC_;
      	MonitorElement* hist_EE_SC_NumBC_;

      	double hist_min_NumBC_;
      	double hist_max_NumBC_;
      	int    hist_bins_NumBC_;

      	MonitorElement* hist_EB_SC_ET_;
      	MonitorElement* hist_EE_SC_ET_;

      	double hist_min_ET_;
      	double hist_max_ET_;
      	int    hist_bins_ET_;

      	MonitorElement* hist_EB_SC_Eta_;
      	MonitorElement* hist_EE_SC_Eta_;

      	double hist_min_Eta_;
      	double hist_max_Eta_;
      	int    hist_bins_Eta_;

      	MonitorElement* hist_EB_SC_Phi_;
      	MonitorElement* hist_EE_SC_Phi_;

      	double hist_min_Phi_;
      	double hist_max_Phi_;
      	int    hist_bins_Phi_;

      	MonitorElement* hist_EB_SC_S1toS9_;
      	MonitorElement* hist_EE_SC_S1toS9_;

      	double hist_min_S1toS9_;
      	double hist_max_S1toS9_;
      	int    hist_bins_S1toS9_;

      	MonitorElement* hist_EB_SC_S25toE_;
      	MonitorElement* hist_EE_SC_S25toE_;

      	double hist_min_S25toE_;
      	double hist_max_S25toE_;
      	int    hist_bins_S25toE_;

      	MonitorElement* hist_EB_SC_EToverTruth_;
      	MonitorElement* hist_EE_SC_EToverTruth_;

      	double hist_min_EToverTruth_;
      	double hist_max_EToverTruth_;
      	int    hist_bins_EToverTruth_;

      	MonitorElement* hist_EB_SC_deltaEta_;
      	MonitorElement* hist_EE_SC_deltaEta_;

      	double hist_min_deltaEta_;
      	double hist_max_deltaEta_;
      	int    hist_bins_deltaEta_;

      	float ecalEta(float EtaParticle , float Zvertex, float plane_Radius);
};
#endif
