#include "Validation/HcalHits/interface/HcalSimHitsValidation.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"


HcalSimHitsValidation::HcalSimHitsValidation(edm::ParameterSet const& conf) {
  // DQM ROOT output
  outputFile_ = conf.getUntrackedParameter<std::string>("outputFile", "myfile.root");
  
  if ( outputFile_.size() != 0 ) {
    edm::LogInfo("OutputInfo") << " Hcal SimHit Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("OutputInfo") << " Hcal SimHit Task histograms will NOT be saved";
  }
  
  nevtot = 0;
  
  dbe_ = 0;
  // get hold of back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();
   
  Char_t histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("HcalSimHitsV/HcalSimHitTask");

    // General counters
    sprintf  (histo, "N_HB" );
    Nhb = dbe_->book1D(histo, histo, 2600,0.,2600.);
    sprintf  (histo, "N_HE" );
    Nhe = dbe_->book1D(histo, histo, 2600,0.,2600.);
    sprintf  (histo, "N_HO" );
    Nho = dbe_->book1D(histo, histo, 2200,0.,2200.);
    sprintf  (histo, "N_HF" );
    Nhf = dbe_->book1D(histo, histo, 1800,0., 1800.);

    sprintf  (histo, "Energy_vs_ieta_HB1" );
    Energy_vs_ieta_HB1 = dbe_->book1D(histo, histo, 82, -41., 41.);
    sprintf  (histo, "Energy_vs_ieta_HB2" );
    Energy_vs_ieta_HB2 = dbe_->book1D(histo, histo, 82, -41., 41.);
    sprintf  (histo, "Energy_vs_ieta_HE1" );
    Energy_vs_ieta_HE1 = dbe_->book1D(histo, histo, 82, -41., 41.);
    sprintf  (histo, "Energy_vs_ieta_HE2" );
    Energy_vs_ieta_HE2 = dbe_->book1D(histo, histo, 82, -41., 41.);
    sprintf  (histo, "Energy_vs_ieta_HE3" );
    Energy_vs_ieta_HE3 = dbe_->book1D(histo, histo, 82, -41., 41.);
    sprintf  (histo, "Energy_vs_ieta_HO" );
    Energy_vs_ieta_HO = dbe_->book1D(histo, histo, 82, -41., 41.);
    sprintf  (histo, "Energy_vs_ieta_HF1" );
    Energy_vs_ieta_HF1 = dbe_->book1D(histo, histo, 82, -41., 41.);
    sprintf  (histo, "Energy_vs_ieta_HF2" );
    Energy_vs_ieta_HF2 = dbe_->book1D(histo, histo, 82, -41., 41.);

    sprintf (histo, "HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths");
    meEnConeEtaProfile = dbe_->bookProfile(histo, histo, 82, -41., 41., 210, -10., 200.);  
    
    sprintf (histo, "HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_E");
    meEnConeEtaProfile_E = dbe_->bookProfile(histo, histo, 82, -41., 41., 210, -10., 200.);  
    
    
    sprintf (histo, "HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_EH");
    meEnConeEtaProfile_EH = dbe_->bookProfile(histo, histo, 82, -41., 41., 210, -10., 200.);  
    
   }  //end-of if(_dbe) 

}


HcalSimHitsValidation::~HcalSimHitsValidation() { }

void HcalSimHitsValidation::endJob() { 
  //before check that histos are there....

  // check if ME still there (and not killed by MEtoEDM for memory saving)
  if( dbe_ )
    {
      // check existence of first histo in the list
      if (! dbe_->get("HcalRecHitsV/HcalRecHitTask/N_HB")) return;
    }
  else
    return;
  
  //======================================  
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


void HcalSimHitsValidation::beginJob(){ }

void HcalSimHitsValidation::analyze(edm::Event const& ev, edm::EventSetup const& c) {

  using namespace edm;
  using namespace std;

  cout<<nevtot<<endl;

  //===========================================================================
  // Getting SimHits
  //===========================================================================

  double phi_MC = -999.;  // phi of initial particle from HepMC
  double eta_MC = -999.;  // eta of initial particle from HepMC

  edm::Handle<edm::HepMCProduct> evtMC;
  //  ev.getByLabel("VtxSmeared",evtMC);
  ev.getByLabel("generator",evtMC);  // generator in late 310_preX
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
    //    phi_MC = phip;
    //    eta_MC = etap;
    double pt  = (*p)->momentum().perp();
    if(pt > maxPt) {npart++; maxPt = pt; phi_MC = phip; eta_MC = etap; }
  }

  double partR   = 0.3;
//   double searchR = 1.0; 

//   double maxES = -9999.;
//   double etaHotS = 1000.;
//   double phiHotS = 1000.;
  
  edm::Handle<PCaloHitContainer> hcalHits;
  ev.getByLabel("g4SimHits","HcalHits",hcalHits);
  const PCaloHitContainer * SimHitResult = hcalHits.product () ;
    
//   double enSimHits    = 0.;
//   double enSimHitsHB  = 0.;
//   double enSimHitsHE  = 0.;
//   double enSimHitsHO  = 0.;
//   double enSimHitsHF  = 0.;
//   double enSimHitsHFL = 0.;
//   double enSimHitsHFS = 0.;
  // sum of SimHits in the cone 

  float eta_diff;
  float etaMax  = 9999;
  int ietaMax = 0;

  double HcalCone = 0;

  c.get<CaloGeometryRecord>().get (geometry);
    
  for (std::vector<PCaloHit>::const_iterator SimHits = SimHitResult->begin () ; SimHits != SimHitResult->end(); ++SimHits) {
    HcalDetId cell(SimHits->id());
    int sub =  cell.subdet();
    const CaloCellGeometry* cellGeometry = geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
    double etaS = cellGeometry->getPosition().eta () ;
    double phiS = cellGeometry->getPosition().phi () ;
    double en   = SimHits->energy();    
   
    double r  = dR(eta_MC, phi_MC, etaS, phiS);
       
    if ( r < partR ){ // just energy in the small cone
      // alternative: ietamax -> closest to MC eta  !!!
      eta_diff = fabs(eta_MC - etaS);
      if(eta_diff < etaMax) {
	etaMax  = eta_diff; 
	ietaMax = cell.ieta(); 
      }
      HcalCone += en;
    }
  }
  cout<<HcalCone<<endl;

  meEnConeEtaProfile       ->Fill(double(ietaMax),  HcalCone);    
//   meEnConeEtaProfile_E     ->Fill(double(ietaMax), eEcalCone);   
//   meEnConeEtaProfile_EH    ->Fill(double(ietaMax), HcalCone+eEcalCone); 

  
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


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalSimHitsValidation);

