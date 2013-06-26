#include "Validation/EcalClusters/interface/EgammaSuperClusters.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DQMServices/Core/interface/DQMStore.h"

EgammaSuperClusters::EgammaSuperClusters( const edm::ParameterSet& ps )
{
	outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "");
	//CMSSW_Version_ = ps.getUntrackedParameter<std::string>("CMSSW_Version", "");

	verboseDBE_ = ps.getUntrackedParameter<bool>("verboseDBE", false);

	hist_min_Size_ = ps.getParameter<double>("hist_min_Size");
	hist_max_Size_ = ps.getParameter<double>("hist_max_Size");
	hist_bins_Size_ = ps.getParameter<int>   ("hist_bins_Size");

	hist_min_NumBC_ = ps.getParameter<double>("hist_min_NumBC");
	hist_max_NumBC_ = ps.getParameter<double>("hist_max_NumBC");
	hist_bins_NumBC_ = ps.getParameter<int>   ("hist_bins_NumBC");

	hist_min_ET_ = ps.getParameter<double>("hist_min_ET");
	hist_max_ET_ = ps.getParameter<double>("hist_max_ET");
	hist_bins_ET_ = ps.getParameter<int>   ("hist_bins_ET");

	hist_min_Eta_ = ps.getParameter<double>("hist_min_Eta");
	hist_max_Eta_ = ps.getParameter<double>("hist_max_Eta");
	hist_bins_Eta_ = ps.getParameter<int>   ("hist_bins_Eta");

	hist_min_Phi_ = ps.getParameter<double>("hist_min_Phi");
	hist_max_Phi_ = ps.getParameter<double>("hist_max_Phi");
	hist_bins_Phi_ = ps.getParameter<int>   ("hist_bins_Phi");

	hist_min_S1toS9_ = ps.getParameter<double>("hist_min_S1toS9");
	hist_max_S1toS9_ = ps.getParameter<double>("hist_max_S1toS9");
	hist_bins_S1toS9_ = ps.getParameter<int>   ("hist_bins_S1toS9");

	hist_min_S25toE_ = ps.getParameter<double>("hist_min_S25toE");
	hist_max_S25toE_ = ps.getParameter<double>("hist_max_S25toE");
	hist_bins_S25toE_ = ps.getParameter<int>   ("hist_bins_S25toE");

	hist_min_EoverTruth_ = ps.getParameter<double>("hist_min_EoverTruth");
	hist_max_EoverTruth_ = ps.getParameter<double>("hist_max_EoverTruth");
	hist_bins_EoverTruth_ = ps.getParameter<int>   ("hist_bins_EoverTruth");
	
	hist_min_deltaR_ = ps.getParameter<double>("hist_min_deltaR");
	hist_max_deltaR_ = ps.getParameter<double>("hist_max_deltaR");
	hist_bins_deltaR_ = ps.getParameter<int>   ("hist_bins_deltaR");

	hist_min_phiWidth_ = ps.getParameter<double>("hist_min_phiWidth");
	hist_max_phiWidth_ = ps.getParameter<double>("hist_max_phiWidth");
	hist_bins_phiWidth_ = ps.getParameter<int>("hist_bins_phiWidth");

	hist_min_etaWidth_ = ps.getParameter<double>("hist_min_etaWidth");
	hist_max_etaWidth_ = ps.getParameter<double>("hist_max_etaWidth");
	hist_bins_etaWidth_ = ps.getParameter<int>("hist_bins_etaWidth");

	hist_bins_preshowerE_ = ps.getParameter<int>("hist_bins_preshowerE");
	hist_min_preshowerE_ = ps.getParameter<double>("hist_min_preshowerE");
	hist_max_preshowerE_ = ps.getParameter<double>("hist_max_preshowerE");

	hist_min_R_ = ps.getParameter<double>("hist_min_R");
	hist_max_R_ = ps.getParameter<double>("hist_max_R");
	hist_bins_R_ = ps.getParameter<int>   ("hist_bins_R");

	MCTruthCollection_ = ps.getParameter<edm::InputTag>("MCTruthCollection");

	barrelRawSuperClusterCollection_ = ps.getParameter<edm::InputTag>("barrelRawSuperClusterCollection");
        barrelCorSuperClusterCollection_ = ps.getParameter<edm::InputTag>("barrelCorSuperClusterCollection");

  	endcapRawSuperClusterCollection_ = ps.getParameter<edm::InputTag>("endcapRawSuperClusterCollection");
        endcapPreSuperClusterCollection_ = ps.getParameter<edm::InputTag>("endcapPreSuperClusterCollection");
        endcapCorSuperClusterCollection_ = ps.getParameter<edm::InputTag>("endcapCorSuperClusterCollection");

        barrelRecHitCollection_ = ps.getParameter<edm::InputTag>("barrelRecHitCollection");
        endcapRecHitCollection_ = ps.getParameter<edm::InputTag>("endcapRecHitCollection");
}

EgammaSuperClusters::~EgammaSuperClusters() {}

void EgammaSuperClusters::beginJob() 
{
  	dbe_ = edm::Service<DQMStore>().operator->();                   

  	if ( verboseDBE_ )
	{
  	dbe_->setVerbose(1);
		dbe_->showDirStructure();
	}
	else 
		dbe_->setVerbose(0);

	//dbe_->setCurrentFolder("Ecal/CMSSW_"+CMSSW_Version_+"/EcalClusters/SuperClusters/");
	dbe_->setCurrentFolder("EcalClusterV/EcalSuperClusters/");

	// Number of SuperClusters
	//
	hist_EB_RawSC_Size_ 
		= dbe_->book1D("hist_EB_RawSC_Size_","# Raw SuperClusters in Barrel",
			hist_bins_Size_,hist_min_Size_,hist_max_Size_);
  	hist_EE_RawSC_Size_ 
		= dbe_->book1D("hist_EE_RawSC_Size_","# Raw SuperClusters in Endcap",
			hist_bins_Size_,hist_min_Size_,hist_max_Size_);
        hist_EB_CorSC_Size_
                = dbe_->book1D("hist_EB_CorSC_Size_","# Corrected SuperClusters in Barrel",
                        hist_bins_Size_,hist_min_Size_,hist_max_Size_);
        hist_EE_CorSC_Size_
                = dbe_->book1D("hist_EE_CorSC_Size_","# Corrected SuperClusters in Endcap",
                        hist_bins_Size_,hist_min_Size_,hist_max_Size_);
        hist_EE_PreSC_Size_
                = dbe_->book1D("hist_EE_PreSC_Size_","# SuperClusters with Preshower in Endcap",
                        hist_bins_Size_,hist_min_Size_,hist_max_Size_);

	// Number of BasicClusters in SuperCluster
	//
	hist_EB_RawSC_NumBC_ 
		= dbe_->book1D("hist_EB_RawSC_NumBC_","# of Basic Clusters in Raw Super Clusters in Barrel",
			hist_bins_NumBC_,hist_min_NumBC_,hist_max_NumBC_);
  	hist_EE_RawSC_NumBC_ 
		= dbe_->book1D("hist_EE_RawSC_NumBC_","# of Basic Clusters in Raw Super Clusters in Endcap",
		hist_bins_NumBC_,hist_min_NumBC_,hist_max_NumBC_);
        hist_EB_CorSC_NumBC_
                = dbe_->book1D("hist_EB_CorSC_NumBC_","# of Basic Clusters in Corrected SuperClusters in Barrel",
                        hist_bins_NumBC_,hist_min_NumBC_,hist_max_NumBC_);
        hist_EE_CorSC_NumBC_
                = dbe_->book1D("hist_EE_CorSC_NumBC_","# of Basic Clusters in Corrected SuperClusters in Endcap",
                        hist_bins_NumBC_,hist_min_NumBC_,hist_max_NumBC_);
        hist_EE_PreSC_NumBC_
                = dbe_->book1D("hist_EE_PreSC_NumBC_","# of Basic Clusters in SuperClusters with Preshower in Endcap",
                        hist_bins_NumBC_,hist_min_NumBC_,hist_max_NumBC_);

	// ET distribution of SuperClusters
	//
  	hist_EB_RawSC_ET_ 
		= dbe_->book1D("hist_EB_RawSC_ET_","ET of Raw SuperClusters in Barrel",
			hist_bins_ET_,hist_min_ET_,hist_max_ET_);
  	hist_EE_RawSC_ET_ 
		= dbe_->book1D("hist_EE_RawSC_ET_","ET of Raw SuperClusters in Endcap",
			hist_bins_ET_,hist_min_ET_,hist_max_ET_);
        hist_EB_CorSC_ET_
                = dbe_->book1D("hist_EB_CorSC_ET_","ET of Corrected SuperClusters in Barrel",
                        hist_bins_ET_,hist_min_ET_,hist_max_ET_);
        hist_EE_CorSC_ET_
                = dbe_->book1D("hist_EE_CorSC_ET_","ET of Corrected SuperClusters in Endcap",
                        hist_bins_ET_,hist_min_ET_,hist_max_ET_);
        hist_EE_PreSC_ET_
                = dbe_->book1D("hist_EE_PreSC_ET_","ET of SuperClusters with Preshower in Endcap",
                        hist_bins_ET_,hist_min_ET_,hist_max_ET_);

	// Eta distribution of SuperClusters
	//
  	hist_EB_RawSC_Eta_ 
		= dbe_->book1D("hist_EB_RawSC_Eta_","Eta of Raw SuperClusters in Barrel",
			hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_);
 	hist_EE_RawSC_Eta_ 
		= dbe_->book1D("hist_EE_RawSC_Eta_","Eta of Raw SuperClusters in Endcap",
			hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_);
        hist_EB_CorSC_Eta_
                = dbe_->book1D("hist_EB_CorSC_Eta_","Eta of Corrected SuperClusters in Barrel",
                        hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_);
        hist_EE_CorSC_Eta_
                = dbe_->book1D("hist_EE_CorSC_Eta_","Eta of Corrected SuperClusters in Endcap",
                        hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_);
        hist_EE_PreSC_Eta_
                = dbe_->book1D("hist_EE_PreSC_Eta_","Eta of SuperClusters with Preshower in Endcap",
                        hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_);

        // Phi distribution of SuperClusters
        //
        hist_EB_RawSC_Phi_
                = dbe_->book1D("hist_EB_RawSC_Phi_","Phi of Raw SuperClusters in Barrel",
                        hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_);
        hist_EE_RawSC_Phi_
                = dbe_->book1D("hist_EE_RawSC_Phi_","Phi of Raw SuperClusters in Endcap",
                        hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_);
        hist_EB_CorSC_Phi_
                = dbe_->book1D("hist_EB_CorSC_Phi_","Phi of Corrected SuperClusters in Barrel",
                        hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_);
        hist_EE_CorSC_Phi_
                = dbe_->book1D("hist_EE_CorSC_Phi_","Phi of Corrected SuperClusters in Endcap",
                        hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_);
        hist_EE_PreSC_Phi_
                = dbe_->book1D("hist_EE_PreSC_Phi_","Phi of SuperClusters with Preshower in Endcap",
                        hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_);

	// S1/S9 distribution of SuperClusters
	//
  	hist_EB_RawSC_S1toS9_ 
		= dbe_->book1D("hist_EB_RawSC_S1toS9_","S1/S9 of Raw Super Clusters in Barrel",
			hist_bins_S1toS9_,hist_min_S1toS9_,hist_max_S1toS9_);
  	hist_EE_RawSC_S1toS9_ 
		= dbe_->book1D("hist_EE_RawSC_S1toS9_","S1/S9 of Raw Super Clusters in Endcap",
			hist_bins_S1toS9_,hist_min_S1toS9_,hist_max_S1toS9_);
        hist_EB_CorSC_S1toS9_
                = dbe_->book1D("hist_EB_CorSC_S1toS9_","S1/S9 of Corrected SuperClusters in Barrel",
                        hist_bins_S1toS9_,hist_min_S1toS9_,hist_max_S1toS9_);
        hist_EE_CorSC_S1toS9_
                = dbe_->book1D("hist_EE_CorSC_S1toS9_","S1/S9 of Corrected SuperClusters in Endcap",
                        hist_bins_S1toS9_,hist_min_S1toS9_,hist_max_S1toS9_);
        hist_EE_PreSC_S1toS9_
                = dbe_->book1D("hist_EE_PreSC_S1toS9_","S1/S9 of SuperClusters with Preshower in Endcap",
                        hist_bins_S1toS9_,hist_min_S1toS9_,hist_max_S1toS9_);

        // S25/E distribution of SuperClusters
        //
  	hist_EB_RawSC_S25toE_ 
		= dbe_->book1D("hist_EB_RawSC_S25toE_","S25/E of Raw Super Clusters in Barrel",
			hist_bins_S25toE_,hist_min_S25toE_,hist_max_S25toE_);
  	hist_EE_RawSC_S25toE_ 
		= dbe_->book1D("hist_EE_RawSC_S25toE_","S25/E of Raw Super Clusters in Endcap",
			hist_bins_S25toE_,hist_min_S25toE_,hist_max_S25toE_);
        hist_EB_CorSC_S25toE_
                = dbe_->book1D("hist_EB_CorSC_S25toE_","S25/E of Corrected SuperClusters in Barrel",
                        hist_bins_S25toE_,hist_min_S25toE_,hist_max_S25toE_);
        hist_EE_CorSC_S25toE_
                = dbe_->book1D("hist_EE_CorSC_S25toE_","S25/E of Corrected SuperClusters in Endcap",
                        hist_bins_S25toE_,hist_min_S25toE_,hist_max_S25toE_);
        hist_EE_PreSC_S25toE_
                = dbe_->book1D("hist_EE_PreSC_S25toE_","S25/E of SuperClusters with Preshower in Endcap",
                        hist_bins_S25toE_,hist_min_S25toE_,hist_max_S25toE_);

	// E/E(true) distribution of SuperClusters
	//
  	hist_EB_RawSC_EoverTruth_ 
		= dbe_->book1D("hist_EB_RawSC_EoverTruth_","E/True E of Raw SuperClusters in Barrel",	
			hist_bins_EoverTruth_,hist_min_EoverTruth_,hist_max_EoverTruth_);
  	hist_EE_RawSC_EoverTruth_ 
		= dbe_->book1D("hist_EE_RawSC_EoverTruth_","E/True E of Raw SuperClusters in Endcap",
			hist_bins_EoverTruth_,hist_min_EoverTruth_,hist_max_EoverTruth_);
        hist_EB_CorSC_EoverTruth_
                = dbe_->book1D("hist_EB_CorSC_EoverTruth_","E/True E of Corrected SuperClusters in Barrel",
                        hist_bins_EoverTruth_,hist_min_EoverTruth_,hist_max_EoverTruth_);
        hist_EE_CorSC_EoverTruth_
                = dbe_->book1D("hist_EE_CorSC_EoverTruth_","E/True E of Corrected SuperClusters in Endcap",
                        hist_bins_EoverTruth_,hist_min_EoverTruth_,hist_max_EoverTruth_);
        hist_EE_PreSC_EoverTruth_
                = dbe_->book1D("hist_EE_PreSC_EoverTruth_","E/True E of SuperClusters with Preshower in Endcap",
                        hist_bins_EoverTruth_,hist_min_EoverTruth_,hist_max_EoverTruth_);

	// dR distribution of SuperClusters from truth
	//
  	hist_EB_RawSC_deltaR_ 
		= dbe_->book1D("hist_EB_RawSC_deltaR_","dR to MC truth of Raw Super Clusters in Barrel",
			hist_bins_deltaR_,hist_min_deltaR_,hist_max_deltaR_);
  	hist_EE_RawSC_deltaR_ 
		= dbe_->book1D("hist_EE_RawSC_deltaR_","dR to MC truth of Raw Super Clusters in Endcap",
			hist_bins_deltaR_,hist_min_deltaR_,hist_max_deltaR_);
        hist_EB_CorSC_deltaR_
                = dbe_->book1D("hist_EB_CorSC_deltaR_","dR to MC truth of Corrected SuperClusters in Barrel",
                        hist_bins_deltaR_,hist_min_deltaR_,hist_max_deltaR_);
        hist_EE_CorSC_deltaR_
                = dbe_->book1D("hist_EE_CorSC_deltaR_","dR to MC truth of Corrected SuperClusters in Endcap",
                        hist_bins_deltaR_,hist_min_deltaR_,hist_max_deltaR_);
        hist_EE_PreSC_deltaR_
                = dbe_->book1D("hist_EE_PreSC_deltaR_","dR to MC truth of SuperClusters with Preshower in Endcap",
                        hist_bins_deltaR_,hist_min_deltaR_,hist_max_deltaR_);

	// phi width stored in corrected SuperClusters
	hist_EB_CorSC_phiWidth_
                = dbe_->book1D("hist_EB_CorSC_phiWidth_","phiWidth of Corrected Super Clusters in Barrel",
                        hist_bins_phiWidth_,hist_min_phiWidth_,hist_max_phiWidth_);
        hist_EE_CorSC_phiWidth_
                = dbe_->book1D("hist_EE_CorSC_phiWidth_","phiWidth of Corrected Super Clusters in Endcap",
                        hist_bins_phiWidth_,hist_min_phiWidth_,hist_max_phiWidth_);

	// eta width stored in corrected SuperClusters
        hist_EB_CorSC_etaWidth_
                = dbe_->book1D("hist_EB_CorSC_etaWidth_","etaWidth of Corrected Super Clusters in Barrel",
                        hist_bins_etaWidth_,hist_min_etaWidth_,hist_max_etaWidth_);
        hist_EE_CorSC_etaWidth_
                = dbe_->book1D("hist_EE_CorSC_etaWidth_","etaWidth of Corrected Super Clusters in Endcap",
                        hist_bins_etaWidth_,hist_min_etaWidth_,hist_max_etaWidth_);


	// preshower energy
	hist_EE_PreSC_preshowerE_
                = dbe_->book1D("hist_EE_PreSC_preshowerE_","preshower energy in Super Clusters with Preshower in Endcap",
                        hist_bins_preshowerE_,hist_min_preshowerE_,hist_max_preshowerE_);
        hist_EE_CorSC_preshowerE_
                = dbe_->book1D("hist_EE_CorSC_preshowerE_","preshower energy in Corrected Super Clusters with Preshower in Endcap",
                        hist_bins_preshowerE_,hist_min_preshowerE_,hist_max_preshowerE_);


	//
	hist_EB_CorSC_ET_vs_Eta_ = dbe_->book2D( "hist_EB_CorSC_ET_vs_Eta_", "Corr Super Cluster ET versus Eta in Barrel", 
					      hist_bins_ET_, hist_min_ET_, hist_max_ET_,
					      hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_ );

	hist_EB_CorSC_ET_vs_Phi_ = dbe_->book2D( "hist_EB_CorSC_ET_vs_Phi_", "Corr Super Cluster ET versus Phi in Barrel", 
					      hist_bins_ET_, hist_min_ET_, hist_max_ET_,
					      hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_ );

	hist_EE_CorSC_ET_vs_Eta_ = dbe_->book2D( "hist_EE_CorSC_ET_vs_Eta_", "Corr Super Cluster ET versus Eta in Endcap", 
					      hist_bins_ET_, hist_min_ET_, hist_max_ET_,
					      hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_ );

	hist_EE_CorSC_ET_vs_Phi_ = dbe_->book2D( "hist_EE_CorSC_ET_vs_Phi_", "Corr Super Cluster ET versus Phi in Endcap", 
					      hist_bins_ET_, hist_min_ET_, hist_max_ET_,
					      hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_ );

	hist_EE_CorSC_ET_vs_R_ = dbe_->book2D( "hist_EE_CorSC_ET_vs_R_", "Corr Super Cluster ET versus Radius in Endcap", 
					      hist_bins_ET_, hist_min_ET_, hist_max_ET_,
					      hist_bins_R_,hist_min_R_,hist_max_R_ );


}

void EgammaSuperClusters::analyze( const edm::Event& evt, const edm::EventSetup& es )
{

	bool skipMC = false;
	bool skipBarrel = false;
	bool skipEndcap = false;

        //
        // Get MCTRUTH
        //
        edm::Handle<edm::HepMCProduct> pMCTruth ;
        evt.getByLabel(MCTruthCollection_, pMCTruth);
        if (!pMCTruth.isValid()) {
          edm::LogError("EgammaSuperClusters") << "Error! can't get collection with label "
                                               << MCTruthCollection_.label();
	  skipMC = true;
        }
	const HepMC::GenEvent* genEvent = pMCTruth->GetEvent();

	if( skipMC ) return;

	//
	// Get the BARREL products 
	//
  	edm::Handle<reco::SuperClusterCollection> pBarrelRawSuperClusters;
	evt.getByLabel(barrelRawSuperClusterCollection_, pBarrelRawSuperClusters);
	if (!pBarrelRawSuperClusters.isValid()) {
	  edm::LogError("EgammaSuperClusters") << "Error! can't get collection with label " 
					       << barrelRawSuperClusterCollection_.label();
	  skipBarrel = true;
  	}

        edm::Handle<reco::SuperClusterCollection> pBarrelCorSuperClusters;
        evt.getByLabel(barrelCorSuperClusterCollection_, pBarrelCorSuperClusters);
        if (!pBarrelCorSuperClusters.isValid()) {
          edm::LogError("EgammaSuperClusters") << "Error! can't get collection with label "
                                               << barrelCorSuperClusterCollection_.label();
	  skipBarrel = true;
        }

	edm::Handle< EBRecHitCollection > pBarrelRecHitCollection;
	evt.getByLabel( barrelRecHitCollection_, pBarrelRecHitCollection );
	if ( ! pBarrelRecHitCollection.isValid() ) {
	  skipBarrel = true;
	}
	edm::Handle< EERecHitCollection > pEndcapRecHitCollection;
	evt.getByLabel( endcapRecHitCollection_, pEndcapRecHitCollection );
	if ( ! pEndcapRecHitCollection.isValid() ) {
	  skipEndcap = true;
	}

	if( skipBarrel || skipEndcap ) return;

	EcalClusterLazyTools lazyTool( evt, es, barrelRecHitCollection_, endcapRecHitCollection_ );

	// Get the BARREL collections        
  	const reco::SuperClusterCollection* barrelRawSuperClusters = pBarrelRawSuperClusters.product();
        const reco::SuperClusterCollection* barrelCorSuperClusters = pBarrelCorSuperClusters.product();

	// Number of entries in collections
  	hist_EB_RawSC_Size_->Fill(barrelRawSuperClusters->size());
        hist_EB_CorSC_Size_->Fill(barrelCorSuperClusters->size());

	// Do RAW BARREL SuperClusters
  	for(reco::SuperClusterCollection::const_iterator aClus = barrelRawSuperClusters->begin(); 
		aClus != barrelRawSuperClusters->end(); aClus++)
	{
		// kinematics
		hist_EB_RawSC_NumBC_->Fill(aClus->clustersSize());
    		hist_EB_RawSC_ET_->Fill(aClus->energy()/std::cosh(aClus->position().eta()));
		hist_EB_RawSC_Eta_->Fill(aClus->position().eta());
		hist_EB_RawSC_Phi_->Fill(aClus->position().phi());

		// cluster shape
                const reco::CaloClusterPtr seed = aClus->seed();
		hist_EB_RawSC_S1toS9_->Fill( lazyTool.eMax( *seed ) / lazyTool.e3x3( *seed ) );
		hist_EB_RawSC_S25toE_->Fill( lazyTool.e5x5( *seed ) / aClus->energy() );

		// truth
		double dRClosest = 999.9;
		double energyClosest = 0;
		closestMCParticle(genEvent, *aClus, dRClosest, energyClosest);

		if (dRClosest < 0.1)
		{
			hist_EB_RawSC_EoverTruth_->Fill(aClus->energy()/energyClosest);		
                        hist_EB_RawSC_deltaR_->Fill(dRClosest);

		}

  	}

	// Do CORRECTED BARREL SuperClusters
        for(reco::SuperClusterCollection::const_iterator aClus = barrelCorSuperClusters->begin();
                aClus != barrelCorSuperClusters->end(); aClus++)
        {
                // kinematics
                hist_EB_CorSC_NumBC_->Fill(aClus->clustersSize());
                hist_EB_CorSC_ET_->Fill(aClus->energy()/std::cosh(aClus->position().eta()));
                hist_EB_CorSC_Eta_->Fill(aClus->position().eta());
                hist_EB_CorSC_Phi_->Fill(aClus->position().phi());

		hist_EB_CorSC_ET_vs_Eta_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), aClus->eta() );
		hist_EB_CorSC_ET_vs_Phi_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), aClus->phi() );


                // cluster shape
                const reco::CaloClusterPtr seed = aClus->seed();
                hist_EB_CorSC_S1toS9_->Fill( lazyTool.eMax( *seed ) / lazyTool.e3x3( *seed ) );
                hist_EB_CorSC_S25toE_->Fill( lazyTool.e5x5( *seed ) / aClus->energy() );

		// correction variables
		hist_EB_CorSC_phiWidth_->Fill(aClus->phiWidth());
                hist_EB_CorSC_etaWidth_->Fill(aClus->etaWidth());

                // truth
                double dRClosest = 999.9;
                double energyClosest = 0;
                closestMCParticle(genEvent, *aClus, dRClosest, energyClosest);

                if (dRClosest < 0.1)
                {
                        hist_EB_CorSC_EoverTruth_->Fill(aClus->energy()/energyClosest);
                        hist_EB_CorSC_deltaR_->Fill(dRClosest);

                }

        }

	//
	// Get the ENDCAP products
	//
  	edm::Handle<reco::SuperClusterCollection> pEndcapRawSuperClusters;
	evt.getByLabel(endcapRawSuperClusterCollection_, pEndcapRawSuperClusters);
	if (!pEndcapRawSuperClusters.isValid()) {
	  edm::LogError("EgammaSuperClusters") << "Error! can't get collection with label " 
					       << endcapRawSuperClusterCollection_.label();
  	}

        edm::Handle<reco::SuperClusterCollection> pEndcapPreSuperClusters;
        evt.getByLabel(endcapPreSuperClusterCollection_, pEndcapPreSuperClusters);
        if (!pEndcapPreSuperClusters.isValid()) {
          edm::LogError("EgammaSuperClusters") << "Error! can't get collection with label "
                                               << endcapPreSuperClusterCollection_.label();
        }

        edm::Handle<reco::SuperClusterCollection> pEndcapCorSuperClusters;
        evt.getByLabel(endcapCorSuperClusterCollection_, pEndcapCorSuperClusters);
        if (!pEndcapCorSuperClusters.isValid()) {
          edm::LogError("EgammaSuperClusters") << "Error! can't get collection with label "
                                               << endcapCorSuperClusterCollection_.label();
        }

	// Get the ENDCAP collections
  	const reco::SuperClusterCollection* endcapRawSuperClusters = pEndcapRawSuperClusters.product();
        const reco::SuperClusterCollection* endcapPreSuperClusters = pEndcapPreSuperClusters.product();
        const reco::SuperClusterCollection* endcapCorSuperClusters = pEndcapCorSuperClusters.product();

	// Number of entries in collections
  	hist_EE_RawSC_Size_->Fill(endcapRawSuperClusters->size());
        hist_EE_PreSC_Size_->Fill(endcapPreSuperClusters->size());
        hist_EE_CorSC_Size_->Fill(endcapCorSuperClusters->size());

	// Do RAW ENDCAP SuperClusters
  	for(reco::SuperClusterCollection::const_iterator aClus = endcapRawSuperClusters->begin(); 
		aClus != endcapRawSuperClusters->end(); aClus++)
	{
		hist_EE_RawSC_NumBC_->Fill(aClus->clustersSize());
    		hist_EE_RawSC_ET_->Fill(aClus->energy()/std::cosh(aClus->position().eta()));
		hist_EE_RawSC_Eta_->Fill(aClus->position().eta());
		hist_EE_RawSC_Phi_->Fill(aClus->position().phi());

                const reco::CaloClusterPtr seed = aClus->seed();
		hist_EE_RawSC_S1toS9_->Fill( lazyTool.eMax( *seed ) / lazyTool.e3x3( *seed ) );
		hist_EE_RawSC_S25toE_->Fill( lazyTool.e5x5( *seed ) / aClus->energy() );

                // truth
                double dRClosest = 999.9;
                double energyClosest = 0;
                closestMCParticle(genEvent, *aClus, dRClosest, energyClosest);

                if (dRClosest < 0.1)
                {
                        hist_EE_RawSC_EoverTruth_->Fill(aClus->energy()/energyClosest);
			hist_EE_RawSC_deltaR_->Fill(dRClosest);
                }

  	}

        // Do ENDCAP SuperClusters with PRESHOWER
        for(reco::SuperClusterCollection::const_iterator aClus = endcapPreSuperClusters->begin();
                aClus != endcapPreSuperClusters->end(); aClus++)
        {
                hist_EE_PreSC_NumBC_->Fill(aClus->clustersSize());
                hist_EE_PreSC_ET_->Fill(aClus->energy()/std::cosh(aClus->position().eta()));
                hist_EE_PreSC_Eta_->Fill(aClus->position().eta());
                hist_EE_PreSC_Phi_->Fill(aClus->position().phi());
		hist_EE_PreSC_preshowerE_->Fill(aClus->preshowerEnergy());

                const reco::CaloClusterPtr seed = aClus->seed();
                hist_EE_PreSC_S1toS9_->Fill( lazyTool.eMax( *seed ) / lazyTool.e3x3( *seed ) );
                hist_EE_PreSC_S25toE_->Fill( lazyTool.e5x5( *seed ) / aClus->energy() );

                // truth
                double dRClosest = 999.9;
                double energyClosest = 0;
                closestMCParticle(genEvent, *aClus, dRClosest, energyClosest);

                if (dRClosest < 0.1)
                {
                        hist_EE_PreSC_EoverTruth_->Fill(aClus->energy()/energyClosest);
                        hist_EE_PreSC_deltaR_->Fill(dRClosest);
                }

        }

        // Do CORRECTED ENDCAP SuperClusters
        for(reco::SuperClusterCollection::const_iterator aClus = endcapCorSuperClusters->begin();
                aClus != endcapCorSuperClusters->end(); aClus++)
        {
                hist_EE_CorSC_NumBC_->Fill(aClus->clustersSize());
                hist_EE_CorSC_ET_->Fill(aClus->energy()/std::cosh(aClus->position().eta()));
                hist_EE_CorSC_Eta_->Fill(aClus->position().eta());
                hist_EE_CorSC_Phi_->Fill(aClus->position().phi());
                hist_EE_CorSC_preshowerE_->Fill(aClus->preshowerEnergy());

		hist_EE_CorSC_ET_vs_Eta_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), aClus->eta() );
		hist_EE_CorSC_ET_vs_Phi_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), aClus->phi() );
		hist_EE_CorSC_ET_vs_R_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), 
					      std::sqrt( std::pow(aClus->x(),2) + std::pow(aClus->y(),2) ) );


                // correction variables
                hist_EE_CorSC_phiWidth_->Fill(aClus->phiWidth());
                hist_EE_CorSC_etaWidth_->Fill(aClus->etaWidth());

                const reco::CaloClusterPtr seed = aClus->seed();
                hist_EE_CorSC_S1toS9_->Fill( lazyTool.eMax( *seed ) / lazyTool.e3x3( *seed ) );
                hist_EE_CorSC_S25toE_->Fill( lazyTool.e5x5( *seed ) / aClus->energy() );

                // truth
                double dRClosest = 999.9;
                double energyClosest = 0;
                closestMCParticle(genEvent, *aClus, dRClosest, energyClosest);

                if (dRClosest < 0.1)
                {
                        hist_EE_CorSC_EoverTruth_->Fill(aClus->energy()/energyClosest);
                        hist_EE_CorSC_deltaR_->Fill(dRClosest);
                }

        }

}

void EgammaSuperClusters::endJob()
{
	if (outputFile_.size() != 0 && dbe_) dbe_->save(outputFile_);
}

//
// Closest MC Particle
//
void EgammaSuperClusters::closestMCParticle(const HepMC::GenEvent *genEvent, const reco::SuperCluster &sc, 
						double &dRClosest, double &energyClosest)
{

	// SuperCluster eta, phi
	double scEta = sc.eta();
 	double scPhi = sc.phi();

	// initialize dRClosest to a large number
	dRClosest = 999.9;

	// loop over the MC truth particles to find the
  	// closest to the superCluster in dR space
        for(HepMC::GenEvent::particle_const_iterator currentParticle = genEvent->particles_begin();
                currentParticle != genEvent->particles_end(); currentParticle++ )
        {
                if((*currentParticle)->status() == 1)
                {
			// need GenParticle in ECAL co-ordinates
                        HepMC::FourVector vtx = (*currentParticle)->production_vertex()->position();
                        double phiTrue = (*currentParticle)->momentum().phi();
                        double etaTrue = ecalEta((*currentParticle)->momentum().eta(), vtx.z()/10., vtx.perp()/10.);

			double dPhi = reco::deltaPhi(phiTrue, scPhi);
			double dEta = scEta - etaTrue;
                        double deltaR = std::sqrt(dPhi*dPhi + dEta*dEta);

                        if(deltaR < dRClosest)
                        {
                        	dRClosest = deltaR;
				energyClosest = (*currentParticle)->momentum().e();
                        }

		} // end if stable particle	

	} // end loop on get particles

}


//
// Compute Eta in the ECAL co-ordinate system
//
float EgammaSuperClusters::ecalEta(float EtaParticle , float Zvertex, float plane_Radius)
{  
	const float R_ECAL           = 136.5;
	const float Z_Endcap         = 328.0;
	const float etaBarrelEndcap  = 1.479;

	if(EtaParticle != 0.)
	{
		float Theta = 0.0  ;
		float ZEcal = (R_ECAL-plane_Radius)*sinh(EtaParticle)+Zvertex;

		if(ZEcal != 0.0) Theta = atan(R_ECAL/ZEcal);
		if(Theta<0.0) Theta = Theta+Geom::pi() ;

		float ETA = - log(tan(0.5*Theta));

		if( fabs(ETA) > etaBarrelEndcap )
		{
			float Zend = Z_Endcap ;
			if(EtaParticle<0.0 )  Zend = -Zend ;
			float Zlen = Zend - Zvertex ;
			float RR = Zlen/sinh(EtaParticle);
			Theta = atan((RR+plane_Radius)/Zend);
			if(Theta<0.0) Theta = Theta+Geom::pi() ;
			ETA = - log(tan(0.5*Theta));
		}
		
		return ETA;
	}
	else
	{
		edm::LogWarning("")  << "[EgammaSuperClusters::ecalEta] Warning: Eta equals to zero, not correcting" ;
		return EtaParticle;
	}
}


