#ifndef Validation_EcalClusters_EgammaSuperClusters_h
#define Validation_EcalClusters_EgammaSuperClusters_h

/**\class EgammaSuperClusters

 Description: SVSuite Super Cluster Validation

 Implementation:
     \\\author: Michael A. Balazs, Nov 2006
*/
//
// $Id: EgammaSuperClusters.h,v 1.13 2009/12/14 22:24:33 wmtan Exp $
//
#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

class EgammaSuperClusters : public edm::EDAnalyzer
{
	public:
      	explicit EgammaSuperClusters( const edm::ParameterSet& );
      	~EgammaSuperClusters();

      	virtual void analyze( const edm::Event&, const edm::EventSetup& );
      	virtual void beginJob();
      	virtual void endJob();

	private:
	std::string outputFile_;
	//std::string CMSSW_Version_;

	bool verboseDBE_;
	DQMStore* dbe_;

	// mc truth
      	edm::InputTag MCTruthCollection_;

	// barrel clusters
      	edm::InputTag barrelRawSuperClusterCollection_;
	edm::InputTag barrelCorSuperClusterCollection_;

	// endcap clusters
      	edm::InputTag endcapRawSuperClusterCollection_;
        edm::InputTag endcapPreSuperClusterCollection_;
        edm::InputTag endcapCorSuperClusterCollection_;

	// collections of hits
      	edm::InputTag barrelRecHitCollection_;
      	edm::InputTag endcapRecHitCollection_;

      	MonitorElement* hist_EB_RawSC_Size_;
      	MonitorElement* hist_EE_RawSC_Size_;
        MonitorElement* hist_EB_CorSC_Size_;
        MonitorElement* hist_EE_CorSC_Size_;
        MonitorElement* hist_EE_PreSC_Size_;
      	double hist_min_Size_;
      	double hist_max_Size_;
      	int    hist_bins_Size_;

      	MonitorElement* hist_EB_RawSC_NumBC_;
      	MonitorElement* hist_EE_RawSC_NumBC_;
        MonitorElement* hist_EB_CorSC_NumBC_;
        MonitorElement* hist_EE_CorSC_NumBC_;
        MonitorElement* hist_EE_PreSC_NumBC_;
      	double hist_min_NumBC_;
      	double hist_max_NumBC_;
      	int    hist_bins_NumBC_;

      	MonitorElement* hist_EB_RawSC_ET_;
      	MonitorElement* hist_EE_RawSC_ET_;
        MonitorElement* hist_EB_CorSC_ET_;
        MonitorElement* hist_EE_CorSC_ET_;
        MonitorElement* hist_EE_PreSC_ET_;
      	double hist_min_ET_;
      	double hist_max_ET_;
      	int    hist_bins_ET_;

      	MonitorElement* hist_EB_RawSC_Eta_;
      	MonitorElement* hist_EE_RawSC_Eta_;
        MonitorElement* hist_EB_CorSC_Eta_;
        MonitorElement* hist_EE_CorSC_Eta_;
        MonitorElement* hist_EE_PreSC_Eta_;
      	double hist_min_Eta_;
      	double hist_max_Eta_;
      	int    hist_bins_Eta_;

      	MonitorElement* hist_EB_RawSC_Phi_;
      	MonitorElement* hist_EE_RawSC_Phi_;
        MonitorElement* hist_EB_CorSC_Phi_;
        MonitorElement* hist_EE_CorSC_Phi_;
        MonitorElement* hist_EE_PreSC_Phi_;
      	double hist_min_Phi_;
      	double hist_max_Phi_;
      	int    hist_bins_Phi_;

      	MonitorElement* hist_EB_RawSC_S1toS9_;
      	MonitorElement* hist_EE_RawSC_S1toS9_;
        MonitorElement* hist_EB_CorSC_S1toS9_;
        MonitorElement* hist_EE_CorSC_S1toS9_;
        MonitorElement* hist_EE_PreSC_S1toS9_;
      	double hist_min_S1toS9_;
      	double hist_max_S1toS9_;
      	int    hist_bins_S1toS9_;

      	MonitorElement* hist_EB_RawSC_S25toE_;
      	MonitorElement* hist_EE_RawSC_S25toE_;
        MonitorElement* hist_EB_CorSC_S25toE_;
        MonitorElement* hist_EE_CorSC_S25toE_;
        MonitorElement* hist_EE_PreSC_S25toE_;
      	double hist_min_S25toE_;
      	double hist_max_S25toE_;
      	int    hist_bins_S25toE_;

      	MonitorElement* hist_EB_RawSC_EoverTruth_;
      	MonitorElement* hist_EE_RawSC_EoverTruth_;
        MonitorElement* hist_EB_CorSC_EoverTruth_;
        MonitorElement* hist_EE_CorSC_EoverTruth_;
        MonitorElement* hist_EE_PreSC_EoverTruth_;
      	double hist_min_EoverTruth_;
      	double hist_max_EoverTruth_;
      	int    hist_bins_EoverTruth_;

      	MonitorElement* hist_EB_RawSC_deltaR_;
      	MonitorElement* hist_EE_RawSC_deltaR_;
        MonitorElement* hist_EB_CorSC_deltaR_;
        MonitorElement* hist_EE_CorSC_deltaR_;
        MonitorElement* hist_EE_PreSC_deltaR_;
      	double hist_min_deltaR_;
      	double hist_max_deltaR_;
      	int    hist_bins_deltaR_;

        MonitorElement* hist_EE_PreSC_preshowerE_;
        MonitorElement* hist_EE_CorSC_preshowerE_;
        double hist_min_preshowerE_;
        double hist_max_preshowerE_;
        int    hist_bins_preshowerE_;

        MonitorElement* hist_EE_CorSC_phiWidth_;
        MonitorElement* hist_EB_CorSC_phiWidth_;
        double hist_min_phiWidth_;
        double hist_max_phiWidth_;
        int    hist_bins_phiWidth_;

        MonitorElement* hist_EE_CorSC_etaWidth_;
        MonitorElement* hist_EB_CorSC_etaWidth_;
        double hist_min_etaWidth_;
        double hist_max_etaWidth_;
        int    hist_bins_etaWidth_;

      	double hist_min_R_;
      	double hist_max_R_;
      	int    hist_bins_R_;

	MonitorElement* hist_EB_CorSC_ET_vs_Eta_;
	MonitorElement* hist_EB_CorSC_ET_vs_Phi_;

	MonitorElement* hist_EE_CorSC_ET_vs_Eta_;
	MonitorElement* hist_EE_CorSC_ET_vs_Phi_;
	MonitorElement* hist_EE_CorSC_ET_vs_R_;


	void closestMCParticle(const HepMC::GenEvent *genEvent, const reco::SuperCluster &sc,
 	                              double &dRClosest, double &energyClosest);


      	float ecalEta(float EtaParticle , float Zvertex, float plane_Radius);

};
#endif

