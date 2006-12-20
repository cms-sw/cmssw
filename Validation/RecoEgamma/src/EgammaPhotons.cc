#include "Validation/RecoEgamma/interface/EgammaPhotons.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include <math.h>

EgammaPhotons::EgammaPhotons( const edm::ParameterSet& ps )
{
  outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "");
  CMSSW_Version_ = ps.getUntrackedParameter<std::string>("CMSSW_Version", "");

  verboseDBE_ = ps.getUntrackedParameter<bool>("verboseDBE", false);

  hist_min_Size_  = ps.getParameter<double>("hist_min_Size");
  hist_max_Size_  = ps.getParameter<double>("hist_max_Size");
  hist_bins_Size_ = ps.getParameter<int>   ("hist_bins_Size");

  hist_min_ET_  = ps.getParameter<double>("hist_min_ET");
  hist_max_ET_  = ps.getParameter<double>("hist_max_ET");
  hist_bins_ET_ = ps.getParameter<int>   ("hist_bins_ET");
	
  hist_min_Eta_  = ps.getParameter<double>("hist_min_Eta");
  hist_max_Eta_  = ps.getParameter<double>("hist_max_Eta");
  hist_bins_Eta_ = ps.getParameter<int>   ("hist_bins_Eta");
	
  hist_min_Phi_  = ps.getParameter<double>("hist_min_Phi");
  hist_max_Phi_  = ps.getParameter<double>("hist_max_Phi");
  hist_bins_Phi_ = ps.getParameter<int>   ("hist_bins_Phi");

  hist_min_EToverTruth_  = ps.getParameter<double>("hist_min_EToverTruth");
  hist_max_EToverTruth_  = ps.getParameter<double>("hist_max_EToverTruth");
  hist_bins_EToverTruth_ = ps.getParameter<int>   ("hist_bins_EToverTruth");
	
  hist_min_deltaEta_  = ps.getParameter<double>("hist_min_deltaEta");
  hist_max_deltaEta_  = ps.getParameter<double>("hist_max_deltaEta");
  hist_bins_deltaEta_ = ps.getParameter<int>   ("hist_bins_deltaEta");
	
  hist_min_deltaPhi_  = ps.getParameter<double>("hist_min_deltaPhi");
  hist_max_deltaPhi_  = ps.getParameter<double>("hist_max_deltaPhi");
  hist_bins_deltaPhi_ = ps.getParameter<int>   ("hist_bins_deltaPhi");

  hist_min_deltaR_  = ps.getParameter<double>("hist_min_deltaR");
  hist_max_deltaR_  = ps.getParameter<double>("hist_max_deltaR");
  hist_bins_deltaR_ = ps.getParameter<int>   ("hist_bins_deltaR");

  hist_min_recoHMass_ = ps.getParameter<double>("hist_min_recoHMass");
  hist_max_recoHMass_ = ps.getParameter<double>("hist_max_recoHMass");
  hist_bins_recoHMass_ = ps.getParameter<int>   ("hist_bins_recoHMass");

  MCTruthCollection_ = ps.getParameter<edm::InputTag>("MCTruthCollection");
  PhotonCollection_  = ps.getParameter<edm::InputTag>("PhotonCollection");
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

  hist_Photon_Size_ 
    = dbe_->book1D("hist_Photon_Size_","# Photons",
      hist_bins_Size_,hist_min_Size_,hist_max_Size_);
  hist_Photon_Barrel_ET_ 
    = dbe_->book1D("hist_Photon_Barrel_ET_","ET of Photons in Barrel",
      hist_bins_ET_,hist_min_ET_,hist_max_ET_);
  hist_Photon_Endcap_ET_ 
    = dbe_->book1D("hist_Photon_Endcap_ET_","ET of Photons in Endcap",
      hist_bins_ET_,hist_min_ET_,hist_max_ET_);
  hist_Photon_Barrel_Eta_ 
    = dbe_->book1D("hist_Photon_Barrel_Eta_","Eta of Photons in Barrel",
      hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_);
  hist_Photon_Endcap_Eta_ 
    = dbe_->book1D("hist_Photon_Endcap_Eta_","Eta of Photons in Endcap",
      hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_);
  hist_Photon_Barrel_Phi_ 
    = dbe_->book1D("hist_Photon_Barrel_Phi_","Phi of Photons in Barrel",
      hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_);
  hist_Photon_Endcap_Phi_ 
    = dbe_->book1D("hist_Photon_Endcap_Phi_","Phi of Photons in Endcap",
      hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_);
  hist_Photon_Barrel_EToverTruth_ 
    = dbe_->book1D("hist_Photon_Barrel_EToverTruth_","ET/True ET of Photons in Barrel",
      hist_bins_EToverTruth_,hist_min_EToverTruth_,hist_max_EToverTruth_);
  hist_Photon_Endcap_EToverTruth_ 
    = dbe_->book1D("hist_Photon_Endcap_EToverTruth_","ET/True ET of Photons in Endcap",
      hist_bins_EToverTruth_,hist_min_EToverTruth_,hist_max_EToverTruth_);
  hist_Photon_Barrel_deltaEta_ 
    = dbe_->book1D("hist_Photon_Barrel_deltaEta_","Eta-True Eta of Photons in Barrel",
      hist_bins_deltaEta_,hist_min_deltaEta_,hist_max_deltaEta_);
  hist_Photon_Endcap_deltaEta_ 
    = dbe_->book1D("hist_Photon_Endcap_deltaEta_","Eta-True Eta of Photons in Endcap",
      hist_bins_deltaEta_,hist_min_deltaEta_,hist_max_deltaEta_);
  hist_Photon_Barrel_deltaPhi_ 
    = dbe_->book1D("hist_Photon_Barrel_deltaPhi_","Phi-True Phi of Photons in Barrel",
      hist_bins_deltaPhi_,hist_min_deltaPhi_,hist_max_deltaPhi_);
  hist_Photon_Endcap_deltaPhi_ 
    = dbe_->book1D("hist_Photon_Endcap_deltaPhi_","Phi-True Phi of Photons in Endcap",
      hist_bins_deltaPhi_,hist_min_deltaPhi_,hist_max_deltaPhi_);
  hist_Photon_Barrel_deltaR_ 
    = dbe_->book1D("hist_Photon_Barrel_deltaR_","delta R of Photons in Barrel",
      hist_bins_deltaR_,hist_min_deltaR_,hist_max_deltaR_);
  hist_Photon_Endcap_deltaR_ 
    = dbe_->book1D("hist_Photon_Endcap_deltaR_","delta R of Photons in Endcap",
      hist_bins_deltaR_,hist_min_deltaR_,hist_max_deltaR_);
  hist_Photon_All_recoHMass_ 
    = dbe_->book1D("hist_Photon_All_recoHMass_","Higgs Mass from Photons in All Regions",
      hist_bins_recoHMass_,hist_min_recoHMass_,hist_max_recoHMass_);    
  hist_Photon_BarrelOnly_recoHMass_ 
    = dbe_->book1D("hist_Photon_BarrelOnly_recoHMass_","Higgs Mass from Photons in Barrel",
      hist_bins_recoHMass_,hist_min_recoHMass_,hist_max_recoHMass_);
  hist_Photon_EndcapOnly_recoHMass_ 
    = dbe_->book1D("hist_Photon_EndcapOnly_recoHMass_","Higgs Mass from Photons in Endcap",
      hist_bins_recoHMass_,hist_min_recoHMass_,hist_max_recoHMass_);
  hist_Photon_Mixed_recoHMass_ 
    = dbe_->book1D("hist_Photon_Mixed_recoHMass_","Higgs Mass from Photons which Split Detectors",
      hist_bins_recoHMass_,hist_min_recoHMass_,hist_max_recoHMass_);  
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

  const reco::PhotonCollection* photons = pPhotons.product();
  std::vector<reco::Photon> photonsMCMatched;

  hist_Photon_Size_->Fill(photons->size());

  for(reco::PhotonCollection::const_iterator aClus = photons->begin(); aClus != photons->end(); aClus++)
  {
    if(std::fabs(aClus->eta()) <= 1.479)
    {
      hist_Photon_Barrel_ET_->Fill(aClus->et());
      hist_Photon_Barrel_Eta_->Fill(aClus->eta());
      hist_Photon_Barrel_Phi_->Fill(aClus->phi());
    }
    else
    {
      hist_Photon_Endcap_ET_->Fill(aClus->et());
      hist_Photon_Endcap_Eta_->Fill(aClus->eta());
      hist_Photon_Endcap_Phi_->Fill(aClus->phi());
    }
  }

  edm::Handle<edm::HepMCProduct> pMCTruth ;
  try
  {
    evt.getByLabel(MCTruthCollection_, pMCTruth);
  }
  catch ( cms::Exception& ex )
  {
    edm::LogError("EgammaPhotons") << "Error! can't get collection with label " << MCTruthCollection_.label();
  }

  const HepMC::GenEvent* genEvent = pMCTruth->GetEvent();
  for(HepMC::GenEvent::particle_const_iterator currentParticle = genEvent->particles_begin(); 
      currentParticle != genEvent->particles_end(); currentParticle++ )
  {
    if(abs((*currentParticle)->pdg_id())==22 && (*currentParticle)->status()==1) 
    {
      double etaTrue = (*currentParticle)->momentum().eta();
      double phiTrue = (*currentParticle)->momentum().phi();
      double etTrue  = (*currentParticle)->momentum().et();

      double etaCurrent, etaFound = 0;
      double phiCurrent, phiFound = 0;
      double etCurrent,  etFound  = 0;

      reco::Photon bestMatchPhoton; 

      double closestParticleDistance = 999; 
		
      for(reco::PhotonCollection::const_iterator aClus = photons->begin(); aClus != photons->end(); aClus++)
      {
        etaCurrent = aClus->eta();
	phiCurrent = aClus->phi();
	etCurrent  = aClus->et();
									
	double deltaR = std::sqrt(std::pow(etaCurrent-etaTrue,2)+std::pow(phiCurrent-phiTrue,2));

	if(deltaR < closestParticleDistance)
	{
	  etFound  = etCurrent;
	  etaFound = etaCurrent;
	  phiFound = phiCurrent;
	  closestParticleDistance = deltaR;
	  bestMatchPhoton = *aClus; 
	}
      }
      
      if(closestParticleDistance < 0.3)
      {
	if(std::fabs(etaFound) <= 1.479)
	{
	  hist_Photon_Barrel_EToverTruth_->Fill(etFound/etTrue);
	  hist_Photon_Barrel_deltaEta_->Fill(etaFound-etaTrue);
	  hist_Photon_Barrel_deltaPhi_->Fill(phiFound-phiTrue);
	  hist_Photon_Barrel_deltaR_->Fill(closestParticleDistance);
	}
	else
	{
	  hist_Photon_Endcap_EToverTruth_->Fill(etFound/etTrue);
	  hist_Photon_Endcap_deltaEta_->Fill(etaFound-etaTrue);
	  hist_Photon_Endcap_deltaPhi_->Fill(phiFound-phiTrue);
	  hist_Photon_Endcap_deltaR_->Fill(closestParticleDistance);
	}

        photonsMCMatched.push_back(bestMatchPhoton);	
      }
    }
  }
  if(photonsMCMatched.size() == 2) findRecoHMass(photonsMCMatched[0], photonsMCMatched[1]);
}

void EgammaPhotons::findRecoHMass(reco::Photon pOne, reco::Photon pTwo)
{

  double cosTheta
    = (cos(pOne.superCluster()->phi() - pTwo.superCluster()->phi()) + sinh(pOne.superCluster()->eta()) * sinh(pTwo.superCluster()->eta())) /
      (cosh(pOne.superCluster()->eta()) * cosh(pTwo.superCluster()->eta()));

  double recoHMass = sqrt(2 * (pOne.superCluster())->energy() * (pTwo.superCluster())->energy() * (1 - cosTheta));

  hist_Photon_All_recoHMass_->Fill(recoHMass);

  if(pOne.superCluster()->eta() < 1.479 && pTwo.superCluster()->eta() < 1.479) 
    hist_Photon_BarrelOnly_recoHMass_->Fill(recoHMass);
  else if(pOne.superCluster()->eta() > 1.479 && pTwo.superCluster()->eta() > 1.479) 
    hist_Photon_EndcapOnly_recoHMass_->Fill(recoHMass);
  else
    hist_Photon_Mixed_recoHMass_->Fill(recoHMass);
}

void EgammaPhotons::endJob()
{
	if (outputFile_.size() != 0 && dbe_) dbe_->save(outputFile_);
}

