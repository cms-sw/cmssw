#include "Validation/RecoEgamma/interface/EgammaPhotons.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"

EgammaPhotons::EgammaPhotons( const edm::ParameterSet& ps )
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

	PhotonCollection_ 								= ps.getParameter<edm::InputTag>("PhotonCollection");
}

EgammaPhotons::~EgammaPhotons() {}

void EgammaPhotons::beginJob(edm::EventSetup const&) 
{
  dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();                   

  if ( verboseDBE_ )
	{
  	dbe_->setVerbose(1);
		dbe_->showDirStructure();
	}
	else 
		dbe_->setVerbose(0);

	dbe_->setCurrentFolder("CMSSW_"+CMSSW_Version_+"/RecoEgamma/Photons/");

	hist_Photon_Size_ 			 							= dbe_->book1D("hist_Photon_Size_","# Photons from Hybrid in Barrel",hist_bins_Size_,hist_min_Size_,hist_max_Size_);
  hist_Photon_ET_ 			 								= dbe_->book1D("hist_Photon_ET_","ET of Photons with Hybrid in Barrel",hist_bins_ET_,hist_min_ET_,hist_max_ET_);
  hist_Photon_Eta_ 			 							  = dbe_->book1D("hist_Photon_Eta_","Eta of Photons with Hybrid in Barrel",hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_);
  hist_Photon_Phi_			 								= dbe_->book1D("hist_Photon_Phi_","Phi of Photons with Hybrid in Barrel",hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_);
}


void EgammaPhotons::analyze( const edm::Event& evt, const edm::EventSetup& es )
{
  edm::Handle<reco::PhotonCollection> pPhotons;
  try
	{
		evt.getByLabel(PhotonCollection_, pPhotons);
  }
	catch ( cms::Exception& ex )
	{
		edm::LogError("EgammaPhotons") << "Error! can't get collection with label " << PhotonCollection_.label();
  }

  const reco::PhotonCollection* Photons = pPhotons.product();
  hist_Photon_Size_->Fill(Photons->size());

  for(reco::PhotonCollection::const_iterator aClus = Photons->begin(); aClus != Photons->end(); aClus++)
	{
    hist_Photon_ET_ 			  				->Fill(aClus->et());
		hist_Photon_Eta_		  					->Fill(aClus->eta());
		hist_Photon_Phi_		 			  		->Fill(aClus->phi());
  }
}

void EgammaPhotons::endJob()
{
	if (outputFile_.size() != 0 && dbe_) dbe_->save(outputFile_);
}
