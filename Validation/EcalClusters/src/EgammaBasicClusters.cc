#include "Validation/EcalClusters/interface/EgammaBasicClusters.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DQMServices/Core/interface/DQMStore.h"

EgammaBasicClusters::EgammaBasicClusters( const edm::ParameterSet& ps )
{
	outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "");
	//CMSSW_Version_ = ps.getUntrackedParameter<std::string>("CMSSW_Version", "");

	verboseDBE_ = ps.getUntrackedParameter<bool>("verboseDBE", false);

	hist_min_Size_ = ps.getParameter<double>("hist_min_Size");
	hist_max_Size_ = ps.getParameter<double>("hist_max_Size");
	hist_bins_Size_ = ps.getParameter<int>   ("hist_bins_Size");

	hist_min_NumRecHits_ = ps.getParameter<double>("hist_min_NumRecHits");
	hist_max_NumRecHits_ = ps.getParameter<double>("hist_max_NumRecHits");
	hist_bins_NumRecHits_ = ps.getParameter<int>   ("hist_bins_NumRecHits");

	hist_min_ET_ = ps.getParameter<double>("hist_min_ET");
	hist_max_ET_ = ps.getParameter<double>("hist_max_ET");
	hist_bins_ET_ = ps.getParameter<int>   ("hist_bins_ET");

	hist_min_Eta_ = ps.getParameter<double>("hist_min_Eta");
	hist_max_Eta_ = ps.getParameter<double>("hist_max_Eta");
	hist_bins_Eta_ = ps.getParameter<int>   ("hist_bins_Eta");

	hist_min_Phi_ = ps.getParameter<double>("hist_min_Phi");
	hist_max_Phi_ = ps.getParameter<double>("hist_max_Phi");
	hist_bins_Phi_ = ps.getParameter<int>   ("hist_bins_Phi");

	hist_min_R_ = ps.getParameter<double>("hist_min_R");
	hist_max_R_ = ps.getParameter<double>("hist_max_R");
	hist_bins_R_ = ps.getParameter<int>   ("hist_bins_R");

	barrelBasicClusterCollection_ = consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("barrelBasicClusterCollection"));
 	endcapBasicClusterCollection_ = consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("endcapBasicClusterCollection"));
}

EgammaBasicClusters::~EgammaBasicClusters() {}

void EgammaBasicClusters::beginJob() 
{
  	dbe_ = edm::Service<DQMStore>().operator->();                   

  	if ( verboseDBE_ )
	{
  		dbe_->setVerbose(1);
		dbe_->showDirStructure();
	}
	else 
		dbe_->setVerbose(0);

	//dbe_->setCurrentFolder("Ecal/CMSSW_"+CMSSW_Version_+"/EcalClusters/BasicClusters/");
	dbe_->setCurrentFolder("EcalClusterV/EcalBasicClusters/");

	hist_EB_BC_Size_ 
		= dbe_->book1D("hist_EB_BC_Size_","# Basic Clusters in Barrel",
			hist_bins_Size_,hist_min_Size_,hist_max_Size_);
  	hist_EE_BC_Size_ 
		= dbe_->book1D("hist_EE_BC_Size_","# Basic Clusters in Endcap",
			hist_bins_Size_,hist_min_Size_,hist_max_Size_);

	hist_EB_BC_NumRecHits_ 
		= dbe_->book1D("hist_EB_BC_NumRecHits_","# of RecHits in Basic Clusters in Barrel",
			hist_bins_NumRecHits_,hist_min_NumRecHits_,hist_max_NumRecHits_);
  	hist_EE_BC_NumRecHits_ 
		= dbe_->book1D("hist_EE_BC_NumRecHits_","# of RecHits in Basic Clusters in Endcap",
			hist_bins_NumRecHits_,hist_min_NumRecHits_,hist_max_NumRecHits_);

  	hist_EB_BC_ET_ 
		= dbe_->book1D("hist_EB_BC_ET_","ET of Basic Clusters in Barrel",
			hist_bins_ET_,hist_min_ET_,hist_max_ET_);
  	hist_EE_BC_ET_ 
		= dbe_->book1D("hist_EE_BC_ET_","ET of Basic Clusters in Endcap",
			hist_bins_ET_,hist_min_ET_,hist_max_ET_);

  	hist_EB_BC_Eta_ 
		= dbe_->book1D("hist_EB_BC_Eta_","Eta of Basic Clusters in Barrel",
			hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_);
  	hist_EE_BC_Eta_ 
		= dbe_->book1D("hist_EE_BC_Eta_","Eta of Basic Clusters in Endcap",
			hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_);

  	hist_EB_BC_Phi_
		= dbe_->book1D("hist_EB_BC_Phi_","Phi of Basic Clusters in Barrel",
			hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_);
  	hist_EE_BC_Phi_ 
		= dbe_->book1D("hist_EE_BC_Phi_","Phi of Basic Clusters in Endcap",
			hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_);

	
	hist_EB_BC_ET_vs_Eta_ = dbe_->book2D( "hist_EB_BC_ET_vs_Eta_", "Basic Cluster ET versus Eta in Barrel", 
					      hist_bins_ET_, hist_min_ET_, hist_max_ET_,
					      hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_ );

	hist_EB_BC_ET_vs_Phi_ = dbe_->book2D( "hist_EB_BC_ET_vs_Phi_", "Basic Cluster ET versus Phi in Barrel", 
					      hist_bins_ET_, hist_min_ET_, hist_max_ET_,
					      hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_ );

	hist_EE_BC_ET_vs_Eta_ = dbe_->book2D( "hist_EE_BC_ET_vs_Eta_", "Basic Cluster ET versus Eta in Endcap", 
					      hist_bins_ET_, hist_min_ET_, hist_max_ET_,
					      hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_ );

	hist_EE_BC_ET_vs_Phi_ = dbe_->book2D( "hist_EE_BC_ET_vs_Phi_", "Basic Cluster ET versus Phi in Endcap", 
					      hist_bins_ET_, hist_min_ET_, hist_max_ET_,
					      hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_ );

	hist_EE_BC_ET_vs_R_ = dbe_->book2D( "hist_EE_BC_ET_vs_R_", "Basic Cluster ET versus Radius in Endcap", 
					      hist_bins_ET_, hist_min_ET_, hist_max_ET_,
					      hist_bins_R_,hist_min_R_,hist_max_R_ );



}

void EgammaBasicClusters::analyze( const edm::Event& evt, const edm::EventSetup& es )
{
  	edm::Handle<reco::BasicClusterCollection> pBarrelBasicClusters;
	evt.getByToken(barrelBasicClusterCollection_, pBarrelBasicClusters);
	if (!pBarrelBasicClusters.isValid()) {

	  Labels l;
	  labelsForToken(barrelBasicClusterCollection_,l);
	  edm::LogError("EgammaBasicClusters") << "Error! can't get collection with label " 
					       << l.module;
	}

  	const reco::BasicClusterCollection* barrelBasicClusters = pBarrelBasicClusters.product();
  	hist_EB_BC_Size_->Fill(barrelBasicClusters->size());

  	for(reco::BasicClusterCollection::const_iterator aClus = barrelBasicClusters->begin(); 
		aClus != barrelBasicClusters->end(); aClus++)
	{
		hist_EB_BC_NumRecHits_->Fill(aClus->size());
    		hist_EB_BC_ET_->Fill(aClus->energy()/std::cosh(aClus->position().eta()));
		hist_EB_BC_Eta_->Fill(aClus->position().eta());
		hist_EB_BC_Phi_->Fill(aClus->position().phi());
		
		hist_EB_BC_ET_vs_Eta_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), aClus->eta() );
		hist_EB_BC_ET_vs_Phi_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), aClus->phi() );
  	}

  	edm::Handle<reco::BasicClusterCollection> pEndcapBasicClusters;

	evt.getByToken(endcapBasicClusterCollection_, pEndcapBasicClusters);
	if (!pEndcapBasicClusters.isValid()) {

	  Labels l;
	  labelsForToken(endcapBasicClusterCollection_,l);
	  edm::LogError("EgammaBasicClusters") << "Error! can't get collection with label " 
					       << l.module;
  	}

  	const reco::BasicClusterCollection* endcapBasicClusters = pEndcapBasicClusters.product();
  	hist_EE_BC_Size_->Fill(endcapBasicClusters->size());

  	for(reco::BasicClusterCollection::const_iterator aClus = endcapBasicClusters->begin(); 
		aClus != endcapBasicClusters->end(); aClus++)
	{
		hist_EE_BC_NumRecHits_->Fill(aClus->size());
    		hist_EE_BC_ET_->Fill(aClus->energy()/std::cosh(aClus->position().eta()));
		hist_EE_BC_Eta_->Fill(aClus->position().eta());
		hist_EE_BC_Phi_->Fill(aClus->position().phi());

		hist_EE_BC_ET_vs_Eta_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), aClus->eta() );
		hist_EE_BC_ET_vs_Phi_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), aClus->phi() );
		hist_EE_BC_ET_vs_R_->Fill( aClus->energy()/std::cosh(aClus->position().eta()), 
					   std::sqrt( std::pow(aClus->x(),2) + std::pow(aClus->y(),2) ) );

  	}
}

void EgammaBasicClusters::endJob()
{
	if (outputFile_.size() != 0 && dbe_) dbe_->save(outputFile_);
}
