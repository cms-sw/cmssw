#include "Validation/EcalClusters/interface/EgammaBasicClusters.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"

EgammaBasicClusters::EgammaBasicClusters( const edm::ParameterSet& ps )
{
	outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "");
	CMSSW_Version_ = ps.getUntrackedParameter<std::string>("CMSSW_Version", "");

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

	hybridBarrelBasicClusterCollection_ = ps.getParameter<edm::InputTag>("hybridBarrelBasicClusterCollection");
 	islandBarrelBasicClusterCollection_ = ps.getParameter<edm::InputTag>("islandBarrelBasicClusterCollection");
 	islandEndcapBasicClusterCollection_ = ps.getParameter<edm::InputTag>("islandEndcapBasicClusterCollection");
}

EgammaBasicClusters::~EgammaBasicClusters() {}

void EgammaBasicClusters::beginJob(edm::EventSetup const&) 
{
  	dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();                   

  	if ( verboseDBE_ )
	{
  		dbe_->setVerbose(1);
		dbe_->showDirStructure();
	}
	else 
		dbe_->setVerbose(0);

	dbe_->setCurrentFolder("CMSSW_"+CMSSW_Version_+"/EcalClusters/BasicClusters/");

	hist_HybridEB_BC_Size_ 
		= dbe_->book1D("hist_HybridEB_BC_Size_","# Basic Clusters from Hybrid in Barrel",
			hist_bins_Size_,hist_min_Size_,hist_max_Size_);
  	hist_IslandEB_BC_Size_ 
		= dbe_->book1D("hist_IslandEB_BC_Size_","# Basic Clusters from Island in Barrel",
			hist_bins_Size_,hist_min_Size_,hist_max_Size_);
  	hist_IslandEE_BC_Size_ 
		= dbe_->book1D("hist_IslandEE_BC_Size_","# Basic Clusters from Island in Endcap",
			hist_bins_Size_,hist_min_Size_,hist_max_Size_);

	hist_HybridEB_BC_NumRecHits_ 
		= dbe_->book1D("hist_HybridEB_BC_NumRecHits_","# of RecHits in Basic Clusters from Hybrid in Barrel",
			hist_bins_NumRecHits_,hist_min_NumRecHits_,hist_max_NumRecHits_);
  	hist_IslandEB_BC_NumRecHits_ 
		= dbe_->book1D("hist_IslandEB_BC_NumRecHits_","# of RecHits in Basic Clusters from Island in Barrel",
			hist_bins_NumRecHits_,hist_min_NumRecHits_,hist_max_NumRecHits_);
  	hist_IslandEE_BC_NumRecHits_ 
		= dbe_->book1D("hist_IslandEE_BC_NumRecHits_","# of RecHits in Basic Clusters from Island in Endcap",
			hist_bins_NumRecHits_,hist_min_NumRecHits_,hist_max_NumRecHits_);

  	hist_HybridEB_BC_ET_ 
		= dbe_->book1D("hist_HybridEB_BC_ET_","ET of Basic Clusters with Hybrid in Barrel",
			hist_bins_ET_,hist_min_ET_,hist_max_ET_);
  	hist_IslandEB_BC_ET_ 
		= dbe_->book1D("hist_IslandEB_BC_ET_","ET of Basic Clusters with Island in Barrel",
			hist_bins_ET_,hist_min_ET_,hist_max_ET_);
  	hist_IslandEE_BC_ET_ 
		= dbe_->book1D("hist_IslandEE_BC_ET_","ET of Basic Clusters with Island in Endcap",
			hist_bins_ET_,hist_min_ET_,hist_max_ET_);

  	hist_HybridEB_BC_Eta_ 
		= dbe_->book1D("hist_HybridEB_BC_Eta_","Eta of Basic Clusters with Hybrid in Barrel",
			hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_);
  	hist_IslandEB_BC_Eta_ 
		= dbe_->book1D("hist_IslandEB_BC_Eta_","Eta of Basic Clusters with Island in Barrel",
			hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_);
  	hist_IslandEE_BC_Eta_ 
		= dbe_->book1D("hist_IslandEE_BC_Eta_","Eta of Basic Clusters with Island in Endcap",
			hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_);

  	hist_HybridEB_BC_Phi_
		= dbe_->book1D("hist_HybridEB_BC_Phi_","Phi of Basic Clusters with Hybrid in Barrel",
			hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_);
  	hist_IslandEB_BC_Phi_ 
		= dbe_->book1D("hist_IslandEB_BC_Phi_","Phi of Basic Clusters with Island in Barrel",
			hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_);
  	hist_IslandEE_BC_Phi_ 
		= dbe_->book1D("hist_IslandEE_BC_Phi_","Phi of Basic Clusters with Island in Endcap",
			hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_);
}

void EgammaBasicClusters::analyze( const edm::Event& evt, const edm::EventSetup& es )
{
  	edm::Handle<reco::BasicClusterCollection> pHybridBarrelBasicClusters;
	try
	{
		evt.getByLabel(hybridBarrelBasicClusterCollection_, pHybridBarrelBasicClusters);
  	}
	catch ( cms::Exception& ex )
	{
		edm::LogError("EgammaBasicClusters") << "Error! can't get collection with label " 
			<< hybridBarrelBasicClusterCollection_.label();
  	}

  	const reco::BasicClusterCollection* hybridBarrelBasicClusters = pHybridBarrelBasicClusters.product();
  	hist_HybridEB_BC_Size_->Fill(hybridBarrelBasicClusters->size());

  	for(reco::BasicClusterCollection::const_iterator aClus = hybridBarrelBasicClusters->begin(); 
		aClus != hybridBarrelBasicClusters->end(); aClus++)
	{
		hist_HybridEB_BC_NumRecHits_->Fill(aClus->getHitsByDetId().size());
    		hist_HybridEB_BC_ET_->Fill(aClus->energy()*aClus->position().theta());
		hist_HybridEB_BC_Eta_->Fill(aClus->position().eta());
		hist_HybridEB_BC_Phi_->Fill(aClus->position().phi());
  	}

  	edm::Handle<reco::BasicClusterCollection> pIslandBarrelBasicClusters;
 	try
	{
		evt.getByLabel(islandBarrelBasicClusterCollection_, pIslandBarrelBasicClusters);
  	}
	catch ( cms::Exception& ex )
	{
		edm::LogError("EgammaBasicClusters") << "Error! can't get collection with label " 
			<< islandBarrelBasicClusterCollection_.label();
  	}

  	const reco::BasicClusterCollection* islandBarrelBasicClusters = pIslandBarrelBasicClusters.product();
  	hist_IslandEB_BC_Size_->Fill(islandBarrelBasicClusters->size());

  	for(reco::BasicClusterCollection::const_iterator aClus = islandBarrelBasicClusters->begin(); 
		aClus != islandBarrelBasicClusters->end(); aClus++)
	{
		hist_IslandEB_BC_NumRecHits_->Fill(aClus->getHitsByDetId().size());
    		hist_IslandEB_BC_ET_->Fill(aClus->energy()*aClus->position().theta());
		hist_IslandEB_BC_Eta_->Fill(aClus->position().eta());
		hist_IslandEB_BC_Phi_->Fill(aClus->position().phi());
  	}

  	edm::Handle<reco::BasicClusterCollection> pIslandEndcapBasicClusters;
  	try
	{
		evt.getByLabel(islandEndcapBasicClusterCollection_, pIslandEndcapBasicClusters);
  	}
	catch ( cms::Exception& ex )
	{
		edm::LogError("EgammaBasicClusters") << "Error! can't get collection with label " 
			<< islandEndcapBasicClusterCollection_.label();
  	}

  	const reco::BasicClusterCollection* islandEndcapBasicClusters = pIslandEndcapBasicClusters.product();
  	hist_IslandEE_BC_Size_->Fill(islandEndcapBasicClusters->size());

  	for(reco::BasicClusterCollection::const_iterator aClus = islandEndcapBasicClusters->begin(); 
		aClus != islandEndcapBasicClusters->end(); aClus++)
	{
		hist_IslandEE_BC_NumRecHits_->Fill(aClus->getHitsByDetId().size());
    		hist_IslandEE_BC_ET_->Fill(aClus->energy()*aClus->position().theta());
		hist_IslandEE_BC_Eta_->Fill(aClus->position().eta());
		hist_IslandEE_BC_Phi_->Fill(aClus->position().phi());
  	}
}

void EgammaBasicClusters::endJob()
{
	if (outputFile_.size() != 0 && dbe_) dbe_->save(outputFile_);
}
