#include "Validation/RecoEgamma/interface/EgammaElectrons.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"

EgammaElectrons::EgammaElectrons( const edm::ParameterSet& ps )
{
	outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "");
	CMSSW_Version_ = ps.getUntrackedParameter<std::string>("CMSSW_Version", "");

  verboseDBE_ = ps.getUntrackedParameter<bool>("verboseDBE", false);

	hist_min_Size_        							= ps.getParameter<double>("hist_min_Size");
	hist_max_Size_										  = ps.getParameter<double>("hist_max_Size");
	hist_bins_Size_ 										= ps.getParameter<int>   ("hist_bins_Size");

	hist_min_ET_   											= ps.getParameter<double>("hist_min_ET");
	hist_max_ET_  										  = ps.getParameter<double>("hist_max_ET");
	hist_bins_ET_ 										  = ps.getParameter<int>   ("hist_bins_ET");
	
	hist_min_Eta_   										= ps.getParameter<double>("hist_min_Eta");
	hist_max_Eta_   										= ps.getParameter<double>("hist_max_Eta");
	hist_bins_Eta_  										= ps.getParameter<int>   ("hist_bins_Eta");
	
	hist_min_Phi_   										= ps.getParameter<double>("hist_min_Phi");
	hist_max_Phi_   										= ps.getParameter<double>("hist_max_Phi");
	hist_bins_Phi_  										= ps.getParameter<int>   ("hist_bins_Phi");

	hist_min_EoverP_   									= ps.getParameter<double>("hist_min_Eta");
	hist_max_EoverP_   									= ps.getParameter<double>("hist_max_Eta");
	hist_bins_EoverP_  									= ps.getParameter<int>   ("hist_bins_Eta");

	hist_min_deltaEtaSCtoTrack_   			= ps.getParameter<double>("hist_min_Phi");
	hist_max_deltaEtaSCtoTrack_   			= ps.getParameter<double>("hist_max_Phi");
	hist_bins_deltaEtaSCtoTrack_  			= ps.getParameter<int>   ("hist_bins_Phi");

	ElectronCollection_ 								= ps.getParameter<edm::InputTag>("ElectronCollection");
}

EgammaElectrons::~EgammaElectrons() {}

void EgammaElectrons::beginJob(edm::EventSetup const&) 
{
  dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();                   

  if ( verboseDBE_ )
	{
  	dbe_->setVerbose(1);
		dbe_->showDirStructure();
	}
	else 
		dbe_->setVerbose(0);

	dbe_->setCurrentFolder("CMSSW_"+CMSSW_Version_+"/RecoEgamma/Electrons/");

	hist_Electron_Size_ 			 							= dbe_->book1D("hist_Electron_Size_","# Electrons from Hybrid in Barrel",hist_bins_Size_,hist_min_Size_,hist_max_Size_);
  hist_Electron_ET_ 			 								= dbe_->book1D("hist_Electron_ET_","ET of Electrons with Hybrid in Barrel",hist_bins_ET_,hist_min_ET_,hist_max_ET_);
  hist_Electron_Eta_ 			 							  = dbe_->book1D("hist_Electron_Eta_","Eta of Electrons with Hybrid in Barrel",hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_);
  hist_Electron_Phi_			 								= dbe_->book1D("hist_Electron_Phi_","Phi of Electrons with Hybrid in Barrel",hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_);
  hist_Electron_EoverP_								    = dbe_->book1D("hist_Electron_EoverP_","E/P of Electrons with Hybrid in Barrel",hist_bins_EoverP_,hist_min_EoverP_,hist_max_EoverP_);
 	hist_Electron_deltaEtaSCtoTrack_			  = dbe_->book1D("hist_Electron_deltaEtaSCtoTrack_","Delta Eta of SC to Tracks of Electrons with Hybrid in Barrel", 	
																													hist_bins_deltaEtaSCtoTrack_,hist_min_deltaEtaSCtoTrack_,hist_max_deltaEtaSCtoTrack_);
}


void EgammaElectrons::analyze( const edm::Event& evt, const edm::EventSetup& es )
{
  edm::Handle<reco::PixelMatchGsfElectronCollection> pElectrons;
  try
	{
		evt.getByLabel(ElectronCollection_, pElectrons);
  }
	catch ( cms::Exception& ex )
	{
		edm::LogError("EgammaElectrons") << "Error! can't get collection with label " << ElectronCollection_.label();
  }

  const reco::PixelMatchGsfElectronCollection* Electrons = pElectrons.product();
  hist_Electron_Size_->Fill(Electrons->size());

  for(reco::PixelMatchGsfElectronCollection::const_iterator aClus = Electrons->begin(); aClus != Electrons->end(); aClus++)
	{
    hist_Electron_ET_ 			  				->Fill(aClus->et());
		hist_Electron_Eta_		  					->Fill(aClus->eta());
		hist_Electron_Phi_		 			  		->Fill(aClus->phi());
		hist_Electron_EoverP_						  ->Fill(aClus->eSuperClusterOverP());
		hist_Electron_deltaEtaSCtoTrack_	->Fill(aClus->deltaEtaSuperClusterTrackAtVtx());
  }
}

void EgammaElectrons::endJob()
{
	if (outputFile_.size() != 0 && dbe_) dbe_->save(outputFile_);
}
