#include "Validation/RecoParticleFlow/plugins/PFClusterValidation.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/ESHandle.h"


PFClusterValidation::PFClusterValidation(const edm::ParameterSet & conf)
{

  tok_evt_ = consumes<edm::HepMCProduct>(edm::InputTag("generatorSmeared"));

  PFClusterECALTok_ = consumes<reco::PFClusterCollection> (conf.getUntrackedParameter<edm::InputTag>("pflowClusterECAL"));  // cms.InputTag("particleFlowClusterECAL");
  PFClusterHCALTok_ = consumes<reco::PFClusterCollection> (conf.getUntrackedParameter<edm::InputTag>("pflowClusterHCAL"));  // cms.InputTag("particleFlowClusterHCAL");
  PFClusterHOTok_ = consumes<reco::PFClusterCollection> (conf.getUntrackedParameter<edm::InputTag>("pflowClusterHO"));  // cms.InputTag("particleFlowClusterECAL"); 
  PFClusterHFTok_ = consumes<reco::PFClusterCollection> (conf.getUntrackedParameter<edm::InputTag>("pflowClusterHF"));  // cms.InputTag("particleFlowClusterECAL"); 

  outputFile_ = conf.getUntrackedParameter<std::string>("outputFile", "myfile.root");
  mc_           = conf.getUntrackedParameter<std::string>("mc", "yes");
  useAllHistos_ = conf.getUntrackedParameter<bool>("useAllHistos", false);

  etaMin[0] = 0.;
  etaMax[0] = 1.4;
  etaMin[1] = 1.4;
  etaMax[1] = 2.9;
  etaMin[2] = 2.9;
  etaMax[2] = 5.2;

  imc = 1;
  if(mc_ == "no") imc = 0;
  
  
  if ( !outputFile_.empty() ) {
    edm::LogInfo("OutputInfo") << " Hcal RecHit Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("OutputInfo") << " Hcal RecHit Task histograms will NOT be saved";
  }
  
  nevent = 0;

}


PFClusterValidation::~PFClusterValidation() {

}


void PFClusterValidation::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & irun, edm::EventSetup const & isetup)
{

  Char_t histo[100];

  ibooker.setCurrentFolder("ParticleFlow/PFClusterV");
  Double_t etaBinsOffset[83] = {-5.191, -4.889, -4.716, -4.538, -4.363, -4.191, -4.013, -3.839, -3.664, -3.489, -3.314, -3.139, -2.964, -2.853, -2.65,
				-2.5, -2.322, -2.172, -2.043, -1.93, -1.83, -1.74, -1.653, -1.566, -1.479, -1.392, -1.305, -1.218, -1.131, -1.044, -0.957,
				-0.879, -0.783, -0.696, -0.609, -0.522, -0.435, -0.348, -0.261, -0.174, -0.087, 0,
				0.087, 0.174, 0.261, 0.348, 0.435, 0.522, 0.609, 0.696, 0.783, 0.879, 0.957, 1.044, 1.131, 1.218, 1.305, 1.392, 1.479,
				1.566, 1.653, 1.74, 1.83, 1.93, 2.043, 2.172, 2.322, 2.5, 2.65, 2.853, 2.964, 3.139, 3.314, 3.489, 3.664, 3.839, 4.013,
				4.191, 4.363, 4.538, 4.716, 4.889, 5.191};

  //These the single pion scan histos
  //-------------------------------------------------------------------------------------------
  sprintf  (histo, "emean_vs_eta_E" );
  emean_vs_eta_E = ibooker.bookProfile(histo, histo, 82, etaBinsOffset , -100., 2000., " ");
  sprintf  (histo, "emean_vs_eta_H" );
  emean_vs_eta_H = ibooker.bookProfile(histo, histo, 82, etaBinsOffset, -100., 2000., " ");
  sprintf  (histo, "emean_vs_eta_EH" );
  emean_vs_eta_EH = ibooker.bookProfile(histo, histo, 82, etaBinsOffset, -100., 2000., " ");
  
  sprintf  (histo, "emean_vs_eta_HF" );
  emean_vs_eta_HF = ibooker.bookProfile(histo, histo, 82, etaBinsOffset, -100., 2000., " ");
  sprintf  (histo, "emean_vs_eta_HO" );
  emean_vs_eta_HO = ibooker.bookProfile(histo, histo, 82, etaBinsOffset, -100., 2000., " ");
  
  sprintf  (histo, "emean_vs_eta_EHF" );
  emean_vs_eta_EHF = ibooker.bookProfile(histo, histo, 82, etaBinsOffset, -100., 2000., " ");
  sprintf  (histo, "emean_vs_eta_EHFO" );
  emean_vs_eta_EHFO = ibooker.bookProfile(histo, histo, 82, etaBinsOffset, -100., 2000., " ");
   
  //-------------------------------------------------------------------------------------------
  
  
} // BOOKING HISTOS


double   phi_MC = 9999.;
double   eta_MC = 9999.;
double partR  = 0.3;
double Rmin   = 9999.;
 
void PFClusterValidation::analyze(edm::Event const& event, edm::EventSetup const& c) {
  
  nevent++;
  
   
  if (imc != 0){
    edm::Handle<edm::HepMCProduct> evtMC;
    event.getByToken(tok_evt_,evtMC);  // generator in late 310_preX
    if (!evtMC.isValid()) {
      std::cout << "no HepMCProduct found" << std::endl;    
    } else {
      // MC=true; // UNUSED
      //    std::cout << "*** source HepMCProduct found"<< std::endl;
    }  
    
    // MC particle with highest pt is taken as a direction reference  
    double maxPt = -99999.;
    int npart    = 0;
    const HepMC::GenEvent * myGenEvent = evtMC->GetEvent();
    for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	  p != myGenEvent->particles_end(); ++p ) {
      double phip = (*p)->momentum().phi();
      double etap = (*p)->momentum().eta();
      double pt  = (*p)->momentum().perp();
      if(pt > maxPt) { npart++; maxPt = pt; phi_MC = phip; eta_MC = etap; }
    }
    //  std::cout << "*** Max pT = " << maxPt <<  std::endl;  
  }    
  
   
  edm::Handle<reco::PFClusterCollection> pfClusterECAL;
  event.getByToken(PFClusterECALTok_, pfClusterECAL);
  reco::PFClusterCollection::const_iterator pf; 
 
  edm::Handle<reco::PFClusterCollection> pfClusterHCAL;
  event.getByToken(PFClusterHCALTok_, pfClusterHCAL);
  
  edm::Handle<reco::PFClusterCollection> pfClusterHO;
  event.getByToken(PFClusterHOTok_, pfClusterHO);
  
  edm::Handle<reco::PFClusterCollection> pfClusterHF;
  event.getByToken(PFClusterHFTok_, pfClusterHF);
    

  double Econe  = 0.;
  double Hcone  = 0.;
  double HFcone  = 0.;
  double HOcone  = 0.;
  
  // sum the energy in a dr cone for each subsystem
  Econe = sumEnergy(pfClusterECAL);
  Hcone = sumEnergy(pfClusterHCAL);
  HOcone = sumEnergy(pfClusterHO);
  HFcone = sumEnergy(pfClusterHF);
  
  //These are the six single pion histos
  emean_vs_eta_E  -> Fill(double(eta_MC), Econe); 
  emean_vs_eta_H  -> Fill(double(eta_MC), Hcone); 
  emean_vs_eta_EH -> Fill(double(eta_MC), Econe+Hcone); 
 
  emean_vs_eta_HF  -> Fill(double(eta_MC), HFcone); 
  emean_vs_eta_HO  -> Fill(double(eta_MC), HOcone); 
  emean_vs_eta_EHF -> Fill(double(eta_MC), Econe+Hcone+HFcone); 
  emean_vs_eta_EHFO -> Fill(double(eta_MC), Econe+Hcone+HFcone+HOcone); 
  
  
} //end for analyze

double PFClusterValidation::dR(double eta1, double phi1, double eta2, double phi2) { 
  double PI = 3.1415926535898;
  double deltaphi= phi1 - phi2;
  if( phi2 > phi1 ) { deltaphi= phi2 - phi1;}
  if(deltaphi > PI) { deltaphi = 2.*PI - deltaphi;}
  double deltaeta = eta2 - eta1;
  double tmp = sqrt(deltaeta* deltaeta + deltaphi*deltaphi);
  return tmp;
}

double PFClusterValidation::sumEnergy(edm::Handle<reco::PFClusterCollection> pfCluster1) {
  reco::PFClusterCollection::const_iterator pf;
  double sumenergy = 0.;
  for ( pf = pfCluster1->begin(); pf != pfCluster1->end(); ++pf ) {
    double eta   = pf->eta();
    double phi   = pf->phi();
    double en     = pf->energy();
    if (imc != 0){
      double r    = dR(eta_MC, phi_MC, eta, phi);
      if( r < partR ){
	sumenergy += en;
      }
    }
  }
  return sumenergy;
}


DEFINE_FWK_MODULE(PFClusterValidation);

