#include "Validation/HcalHits/interface/HcalSimHitsValidation.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"


HcalSimHitsValidation::HcalSimHitsValidation(edm::ParameterSet const& conf) {
  // DQM ROOT output
  outputFile_ = conf.getUntrackedParameter<std::string>("outputFile", "myfile.root");

  // register for data access
  tok_evt_ = consumes<edm::HepMCProduct>(edm::InputTag("generatorSmeared"));
  tok_hcal_ = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits","HcalHits"));
  tok_ecalEB_ = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits","EcalHitsEB"));
  tok_ecalEE_ = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits","EcalHitsEE"));
  
  if ( outputFile_.size() != 0 ) {    edm::LogInfo("OutputInfo") << " Hcal SimHit Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("OutputInfo") << " Hcal SimHit Task histograms will NOT be saved";
  }
  
  nevtot = 0;
  

}


HcalSimHitsValidation::~HcalSimHitsValidation() { }


void HcalSimHitsValidation::bookHistograms(DQMStore::IBooker &ib, edm::Run const &run, edm::EventSetup const &es)
{
  Char_t histo[200];

    ib.setCurrentFolder("HcalSimHitsV/HcalSimHitTask");

    // General counters
    sprintf  (histo, "N_HB" );
    Nhb = ib.book1D(histo, histo, 2600,0.,2600.);
    sprintf  (histo, "N_HE" );
    Nhe = ib.book1D(histo, histo, 2600,0.,2600.);
    sprintf  (histo, "N_HO" );
    Nho = ib.book1D(histo, histo, 2200,0.,2200.);
    sprintf  (histo, "N_HF" );
    Nhf = ib.book1D(histo, histo, 1800,0., 1800.);

    //Mean energy vs iEta TProfiles
    sprintf  (histo, "emean_vs_ieta_HB1" );
    emean_vs_ieta_HB1 = ib.bookProfile(histo, histo, 82, -41., 41., 2010, -10., 2000., "s");
    sprintf  (histo, "emean_vs_ieta_HB2" );
    emean_vs_ieta_HB2 = ib.bookProfile(histo, histo, 82, -41., 41., 2010, -10., 2000., "s");
    sprintf  (histo, "emean_vs_ieta_HE1" );
    emean_vs_ieta_HE1 = ib.bookProfile(histo, histo, 82, -41., 41., 2010, -10. ,2000., "s" );
    sprintf  (histo, "emean_vs_ieta_HE2" );
    emean_vs_ieta_HE2 = ib.bookProfile(histo, histo, 82, -41., 41., 2010, -10., 2000., "s");
    sprintf  (histo, "emean_vs_ieta_HE3" );
    emean_vs_ieta_HE3 = ib.bookProfile(histo, histo, 82, -41., 41., 2010, -10., 2000., "s" );
    sprintf  (histo, "emean_vs_ieta_HO" );
    emean_vs_ieta_HO = ib.bookProfile(histo, histo, 82, -41., 41., 2010, -10., 2000., "s" );
    sprintf  (histo, "emean_vs_ieta_HF1" );
    emean_vs_ieta_HF1 = ib.bookProfile(histo, histo, 82, -41., 41., 2010, -10., 2000., "s" );
    sprintf  (histo, "emean_vs_ieta_HF2" );
    emean_vs_ieta_HF2 = ib.bookProfile(histo, histo, 82, -41., 41., 2010, -10., 2000., "s" );

    //Occupancy vs. iEta TH1Fs
    sprintf  (histo, "occupancy_vs_ieta_HB1" );
    occupancy_vs_ieta_HB1 = ib.book1D(histo, histo, 82, -41., 41.);
    sprintf  (histo, "occupancy_vs_ieta_HB2" );
    occupancy_vs_ieta_HB2 = ib.book1D(histo, histo, 82, -41., 41.);
    sprintf  (histo, "occupancy_vs_ieta_HE1" );
    occupancy_vs_ieta_HE1 = ib.book1D(histo, histo, 82, -41., 41.);
    sprintf  (histo, "occupancy_vs_ieta_HE2" );
    occupancy_vs_ieta_HE2 = ib.book1D(histo, histo, 82, -41., 41.);
    sprintf  (histo, "occupancy_vs_ieta_HE3" );
    occupancy_vs_ieta_HE3 = ib.book1D(histo, histo, 82, -41., 41.);
    sprintf  (histo, "occupancy_vs_ieta_HO" );
    occupancy_vs_ieta_HO = ib.book1D(histo, histo, 82, -41., 41.);
    sprintf  (histo, "occupancy_vs_ieta_HF1" );
    occupancy_vs_ieta_HF1 = ib.book1D(histo, histo, 82, -41., 41.);
    sprintf  (histo, "occupancy_vs_ieta_HF2" );
    occupancy_vs_ieta_HF2 = ib.book1D(histo, histo, 82, -41., 41.);

    //Energy spectra
    sprintf (histo, "HcalSimHitTask_energy_of_simhits_HB" ) ;
    meSimHitsEnergyHB = ib.book1D(histo, histo, 510 , -0.1 , 5.); 

    sprintf (histo, "HcalSimHitTask_energy_of_simhits_HE" ) ;
    meSimHitsEnergyHE = ib.book1D(histo, histo, 510, -0.02, 2.); 

    sprintf (histo, "HcalSimHitTask_energy_of_simhits_HO" ) ;
    meSimHitsEnergyHO = ib.book1D(histo, histo, 510 , -0.1 , 5.); 
    
    sprintf (histo, "HcalSimHitTask_energy_of_simhits_HF" ) ;
    meSimHitsEnergyHF = ib.book1D(histo, histo, 1010 , -5. , 500.); 

    //Energy in Cone
    sprintf (histo, "HcalSimHitTask_En_simhits_cone_profile_vs_ieta_all_depths");
    meEnConeEtaProfile = ib.bookProfile(histo, histo, 82, -41., 41., 210, -10., 200.);  
    
    sprintf (histo, "HcalSimHitTask_En_simhits_cone_profile_vs_ieta_all_depths_E");
    meEnConeEtaProfile_E = ib.bookProfile(histo, histo, 82, -41., 41., 210, -10., 200.);  
      
    sprintf (histo, "HcalSimHitTask_En_simhits_cone_profile_vs_ieta_all_depths_EH");
    meEnConeEtaProfile_EH = ib.bookProfile(histo, histo, 82, -41., 41., 210, -10., 200.);  
    

}


void HcalSimHitsValidation::endJob() { 
  //before check that histos are there....

  // let's see if this breaks anything
  // check if ME still there (and not killed by MEtoEDM for memory saving)
  /*if( dbe_ )
    {
      // check existence of first histo in the list
      if (! dbe_->get("HcalSimHitsV/HcalSimHitTask/N_HB")) return;
    }
  else
    return;*/
  
  //======================================

  for (int i = 1; i <= occupancy_vs_ieta_HB1->getNbinsX(); i++){

    int ieta = i - 42;        // -41 -1, 0 40 
    if(ieta >=0 ) ieta +=1;   // -41 -1, 1 41  - to make it detector-like

    float phi_factor;

    if      (fabs(ieta) <= 20) phi_factor = 72.;
    else if (fabs(ieta) <  40) phi_factor = 36.;
    else                       phi_factor = 18.;
    
    float cnorm;

    //Occupancy vs. iEta TH1Fs
    cnorm = occupancy_vs_ieta_HB1->getBinContent(i) / (phi_factor * nevtot); 
    occupancy_vs_ieta_HB1->setBinContent(i, cnorm);
    cnorm = occupancy_vs_ieta_HB2->getBinContent(i) / (phi_factor * nevtot); 
    occupancy_vs_ieta_HB2->setBinContent(i, cnorm);

    cnorm = occupancy_vs_ieta_HE1->getBinContent(i) / (phi_factor * nevtot); 
    occupancy_vs_ieta_HE1->setBinContent(i, cnorm);
    cnorm = occupancy_vs_ieta_HE2->getBinContent(i) / (phi_factor * nevtot); 
    occupancy_vs_ieta_HE2->setBinContent(i, cnorm);
    cnorm = occupancy_vs_ieta_HE3->getBinContent(i) / (phi_factor * nevtot); 
    occupancy_vs_ieta_HE3->setBinContent(i, cnorm);

    cnorm = occupancy_vs_ieta_HO->getBinContent(i) / (phi_factor * nevtot); 
    occupancy_vs_ieta_HO->setBinContent(i, cnorm);

    cnorm = occupancy_vs_ieta_HF1->getBinContent(i) / (phi_factor * nevtot); 
    occupancy_vs_ieta_HF1->setBinContent(i, cnorm);
    cnorm = occupancy_vs_ieta_HF2->getBinContent(i) / (phi_factor * nevtot); 
    occupancy_vs_ieta_HF2->setBinContent(i, cnorm);
  }


  // let's see if this breaks anything
  //if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


void HcalSimHitsValidation::analyze(edm::Event const& ev, edm::EventSetup const& c) {

  using namespace edm;
  using namespace std;

  //===========================================================================
  // Getting SimHits
  //===========================================================================

  double phi_MC = -999.;  // phi of initial particle from HepMC
  double eta_MC = -999.;  // eta of initial particle from HepMC

  edm::Handle<edm::HepMCProduct> evtMC;
  ev.getByToken(tok_evt_,evtMC);  // generator in late 310_preX
  if (!evtMC.isValid()) {
    std::cout << "no HepMCProduct found" << std::endl;    
  }

  // MC particle with highest pt is taken as a direction reference  
  double maxPt = -99999.;
  int npart    = 0;

  const HepMC::GenEvent * myGenEvent = evtMC->GetEvent();
  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	p != myGenEvent->particles_end(); ++p ) {
    double phip = (*p)->momentum().phi();
    double etap = (*p)->momentum().eta();
    double pt   = (*p)->momentum().perp();
    if(pt > maxPt) {npart++; maxPt = pt; phi_MC = phip; eta_MC = etap; }
  }

  double partR   = 0.3;


  //Hcal SimHits

  //Approximate calibration constants
  const float calib_HB = 120.;
  const float calib_HE = 190.;
  const float calib_HF1 = 1.0/0.383;
  const float calib_HF2 = 1.0/0.368;
  
  edm::Handle<PCaloHitContainer> hcalHits;
  ev.getByToken(tok_hcal_,hcalHits);
  const PCaloHitContainer * SimHitResult = hcalHits.product () ;
    
  float eta_diff;
  float etaMax  = 9999;
  int ietaMax = 0;

  double HcalCone = 0;

  c.get<CaloGeometryRecord>().get (geometry);
    
  for (std::vector<PCaloHit>::const_iterator SimHits = SimHitResult->begin () ; SimHits != SimHitResult->end(); ++SimHits) {
    HcalDetId cell(SimHits->id());
    const CaloCellGeometry* cellGeometry = geometry->getSubdetectorGeometry (cell)->getGeometry (cell);
    double etaS = cellGeometry->getPosition().eta () ;
    double phiS = cellGeometry->getPosition().phi () ;
    double en   = SimHits->energy();    
    
    int sub     = cell.subdet();
    int depth   = cell.depth();
    double ieta = cell.ieta();

    //Energy in Cone 
    double r  = dR(eta_MC, phi_MC, etaS, phiS);
    
    if (r < partR){      
      eta_diff = fabs(eta_MC - etaS);
      if(eta_diff < etaMax) {
	etaMax  = eta_diff; 
	ietaMax = cell.ieta(); 
      }
      //Approximation of calibration
      if      (sub == 1)               HcalCone += en*calib_HB;
      else if (sub == 2)               HcalCone += en*calib_HE;
      else if (sub == 4 && depth == 1) HcalCone += en*calib_HF1;
      else if (sub == 4 && depth == 2) HcalCone += en*calib_HF2;
    }
    
    //Account for lack of ieta = 0
    if (ieta > 0) ieta--;

    //HB
    if (sub == 1){
      meSimHitsEnergyHB->Fill(en);
      if (depth == 1){
	emean_vs_ieta_HB1->Fill(double(ieta), en);
	occupancy_vs_ieta_HB1->Fill(double(ieta));
      }
      if (depth == 2){
	emean_vs_ieta_HB2->Fill(double(ieta), en);
	occupancy_vs_ieta_HB2->Fill(double(ieta));
      }
    }
    //HE
    if (sub == 2){
      meSimHitsEnergyHE->Fill(en);
      if (depth == 1){
	emean_vs_ieta_HE1->Fill(double(ieta), en);
	occupancy_vs_ieta_HE1->Fill(double(ieta));
      }
      if (depth == 2){
	emean_vs_ieta_HE2->Fill(double(ieta), en);
	occupancy_vs_ieta_HE2->Fill(double(ieta));
      }
      if (depth == 3){
	emean_vs_ieta_HE3->Fill(double(ieta), en);
	occupancy_vs_ieta_HE3->Fill(double(ieta));
      }
    }
    //HO
    if (sub == 3){
      meSimHitsEnergyHO->Fill(en);
      emean_vs_ieta_HO->Fill(double(ieta), en);
      occupancy_vs_ieta_HO->Fill(double(ieta));
    }
    //HF
    if (sub == 4){
      meSimHitsEnergyHF->Fill(en);
      if (depth == 1){
	emean_vs_ieta_HF1->Fill(double(ieta), en);
	occupancy_vs_ieta_HF1->Fill(double(ieta));
      }
      if (depth == 2){
	emean_vs_ieta_HF2->Fill(double(ieta), en);
	occupancy_vs_ieta_HF2->Fill(double(ieta));
      }
    }     
  }

  //Ecal EB SimHits
  edm::Handle<PCaloHitContainer> ecalEBHits;
  ev.getByToken(tok_ecalEB_,ecalEBHits);
  const PCaloHitContainer * SimHitResultEB = ecalEBHits.product () ;

  double EcalCone = 0;

  for (std::vector<PCaloHit>::const_iterator SimHits = SimHitResultEB->begin () ; SimHits != SimHitResultEB->end(); ++SimHits) {

    EBDetId EBid = EBDetId(SimHits->id());

    const CaloCellGeometry* cellGeometry = geometry->getSubdetectorGeometry (EBid)->getGeometry (EBid) ;
    double etaS = cellGeometry->getPosition().eta () ;
    double phiS = cellGeometry->getPosition().phi () ;
    double en   = SimHits->energy();    
  
    double r  = dR(eta_MC, phi_MC, etaS, phiS);
    
    if (r < partR) EcalCone += en;   
  }

  //Ecal EE SimHits
  edm::Handle<PCaloHitContainer> ecalEEHits;
  ev.getByToken(tok_ecalEE_,ecalEEHits);
  const PCaloHitContainer * SimHitResultEE = ecalEEHits.product () ;

  for (std::vector<PCaloHit>::const_iterator SimHits = SimHitResultEE->begin () ; SimHits != SimHitResultEE->end(); ++SimHits) {

    EEDetId EEid = EEDetId(SimHits->id());

    const CaloCellGeometry* cellGeometry = geometry->getSubdetectorGeometry (EEid)->getGeometry (EEid) ;
    double etaS = cellGeometry->getPosition().eta () ;
    double phiS = cellGeometry->getPosition().phi () ;
    double en   = SimHits->energy();    
    
    double r  = dR(eta_MC, phi_MC, etaS, phiS);
    
    if (r < partR) EcalCone += en;   
  }

  if (ietaMax != 0){            //If ietaMax == 0, there were no good HCAL SimHits 
    if (ietaMax > 0) ietaMax--; //Account for lack of ieta = 0
    
    meEnConeEtaProfile       ->Fill(double(ietaMax), HcalCone);    
    meEnConeEtaProfile_E     ->Fill(double(ietaMax), EcalCone);
    meEnConeEtaProfile_EH    ->Fill(double(ietaMax), HcalCone+EcalCone); 
  }
  
  nevtot++;
}


double HcalSimHitsValidation::dR(double eta1, double phi1, double eta2, double phi2) { 
  double PI = 3.1415926535898;
  double deltaphi= phi1 - phi2;
  if( phi2 > phi1 ) { deltaphi= phi2 - phi1;}
  if(deltaphi > PI) { deltaphi = 2.*PI - deltaphi;}
  double deltaeta = eta2 - eta1;
  double tmp = sqrt(deltaeta* deltaeta + deltaphi*deltaphi);
  return tmp;
}

double HcalSimHitsValidation::phi12(double phi1, double en1, double phi2, double en2) {
  // weighted mean value of phi1 and phi2
  
  double tmp;
  double PI = 3.1415926535898;
  double a1 = phi1; double a2  = phi2;

  if( a1 > 0.5*PI  && a2 < 0.) a2 += 2*PI; 
  if( a2 > 0.5*PI  && a1 < 0.) a1 += 2*PI; 
  tmp = (a1 * en1 + a2 * en2)/(en1 + en2);
  if(tmp > PI) tmp -= 2.*PI; 
 
  return tmp;

}

double HcalSimHitsValidation::dPhiWsign(double phi1, double phi2) {
  // clockwise      phi2 w.r.t phi1 means "+" phi distance
  // anti-clockwise phi2 w.r.t phi1 means "-" phi distance 

  double PI = 3.1415926535898;
  double a1 = phi1; double a2  = phi2;
  double tmp =  a2 - a1;
  if( a1*a2 < 0.) {
    if(a1 > 0.5 * PI)  tmp += 2.*PI;
    if(a2 > 0.5 * PI)  tmp -= 2.*PI;
  }
  return tmp;

}



DEFINE_FWK_MODULE(HcalSimHitsValidation);

