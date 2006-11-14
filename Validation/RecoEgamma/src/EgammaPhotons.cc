#include "Validation/RecoEgamma/interface/EgammaPhotons.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include <math.h>

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

	hist_min_EToverTruth_   				    = ps.getParameter<double>("hist_min_EToverTruth");
	hist_max_EToverTruth_  							= ps.getParameter<double>("hist_max_EToverTruth");
	hist_bins_EToverTruth_ 							= ps.getParameter<int>   ("hist_bins_EToverTruth");
	
	hist_min_deltaEta_   								= ps.getParameter<double>("hist_min_deltaEta");
	hist_max_deltaEta_   								= ps.getParameter<double>("hist_max_deltaEta");
	hist_bins_deltaEta_  								= ps.getParameter<int>   ("hist_bins_deltaEta");
	
	hist_min_deltaPhi_   								= ps.getParameter<double>("hist_min_deltaPhi");
	hist_max_deltaPhi_   								= ps.getParameter<double>("hist_max_deltaPhi");
	hist_bins_deltaPhi_  								= ps.getParameter<int>   ("hist_bins_deltaPhi");

	hist_min_deltaR_   					  			= ps.getParameter<double>("hist_min_deltaR");
	hist_max_deltaR_   				  				= ps.getParameter<double>("hist_max_deltaR");
	hist_bins_deltaR_  				  				= ps.getParameter<int>   ("hist_bins_deltaR");

	MCTruthCollection_ 							  	= ps.getParameter<edm::InputTag>("MCTruthCollection");
	PhotonCollection_ 						 	  	= ps.getParameter<edm::InputTag>("PhotonCollection");
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
  hist_Photon_EToverTruth_ 			 				= dbe_->book1D("hist_Photon_EToverTruth_","ET/True ET of Photons",hist_bins_EToverTruth_,hist_min_EToverTruth_,hist_max_EToverTruth_);
  hist_Photon_deltaEta_ 			 					= dbe_->book1D("hist_Photon_deltaEta_","Eta-True Eta of Photons",hist_bins_deltaEta_,hist_min_deltaEta_,hist_max_deltaEta_);
  hist_Photon_deltaPhi_			 						= dbe_->book1D("hist_Photon_deltaPhi_","Phi-True Phi of Photons",hist_bins_deltaPhi_,hist_min_deltaPhi_,hist_max_deltaPhi_);
  hist_Photon_deltaR_			 				      = dbe_->book1D("hist_Photon_deltaR_","delta R of Photons",hist_bins_deltaR_,hist_min_deltaR_,hist_max_deltaR_);
}


void EgammaPhotons::analyze( const edm::Event& evt, const edm::EventSetup& es )
{
 	edm::Handle<edm::HepMCProduct> pMCTruth ;
  try
	{
		evt.getByLabel(MCTruthCollection_, pMCTruth);
  }
	catch ( cms::Exception& ex )
	{
		edm::LogError("EgammaPhotons") << "Error! can't get collection with label " << MCTruthCollection_.label();
  }

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
		double etaFound, etaTrue = 0;
		double phiFound, phiTrue = 0;
		double etFound,  etTrue  = 0;

		double closestParticleDistance = 999; 

		etFound  = aClus->et();
		etaFound = aClus->eta();
		phiFound = aClus->phi();

    hist_Photon_ET_ 			  				->Fill(etFound);
		hist_Photon_Eta_		  					->Fill(etaFound);
		hist_Photon_Phi_		 			  		->Fill(phiFound);

		const HepMC::GenEvent* genEvent = pMCTruth->GetEvent();
	  for( HepMC::GenEvent::particle_const_iterator currentParticle = genEvent->particles_begin(); currentParticle != genEvent->particles_end(); currentParticle++ )
	  {
		  if ( abs((*currentParticle)->pdg_id())==22 && (*currentParticle)->status()==1 ) 
			{
			  float et = (*currentParticle)->momentum().et();
			  float eta = (*currentParticle)->momentum().eta();
			  float phi = (*currentParticle)->momentum().phi();

				float deltaR = std::sqrt(std::pow(eta-etaFound,2)+std::pow(phi-phiFound,2)); 

				if(deltaR < closestParticleDistance)
				{
					etTrue  = et;
					etaTrue = eta;
					phiTrue = phi;
					closestParticleDistance = deltaR;
				}
		  }
		}
		
		hist_Photon_EToverTruth_ 				->Fill(etFound/etTrue);
		hist_Photon_deltaEta_		 				->Fill(etaFound-etaTrue);
		hist_Photon_deltaPhi_		 	  		->Fill(phiFound-phiTrue);
		hist_Photon_deltaR_				      ->Fill(closestParticleDistance);
  }
}

void EgammaPhotons::endJob()
{
	if (outputFile_.size() != 0 && dbe_) dbe_->save(outputFile_);
}
