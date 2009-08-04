#include "Validation/CaloTowers/interface/CaloTowersValidation.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

CaloTowersValidation::CaloTowersValidation(edm::ParameterSet const& conf):
  theCaloTowerCollectionLabel(conf.getUntrackedParameter<std::string>("CaloTowerCollectionLabel"))
{
  // DQM ROOT output
  outputFile_ = conf.getUntrackedParameter<std::string>("outputFile", "myfile.root");

  hcalselector_ = conf.getUntrackedParameter<std::string>("hcalselector", "all");

  etaMin[0] = 0.;
  etaMax[0] = 1.4;
  etaMin[1] = 1.4;
  etaMax[1] = 2.9;
  etaMin[2] = 2.9;
  etaMax[2] = 5.2;

  isub = 0;
  if(hcalselector_ == "HB") isub = 1;
  if(hcalselector_ == "HE") isub = 2;
  if(hcalselector_ == "HF") isub = 3;
  
  if ( outputFile_.size() != 0 ) {
    edm::LogInfo("OutputInfo") << " Hcal RecHit Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("OutputInfo") << " Hcal RecHit Task histograms will NOT be saved";
  }
  
  dbe_ = 0;
  // get hold of back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();
   
  // const char * sub = hcalselector_.c_str();

  Char_t histo[100];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("CaloTowersV/CaloTowersTask");
  }


  sprintf  (histo, "Ntowers_per_event_vs_ieta" );
  Ntowers_vs_ieta = dbe_->book1D(histo, histo, 82, -41., 41.);

  sprintf  (histo, "emean_vs_ieta_E" );
  emean_vs_ieta_E = dbe_->bookProfile(histo, histo, 82, -41., 41., 2100, -100., 2000., "s");
  sprintf  (histo, "emean_vs_ieta_H" );
  emean_vs_ieta_H = dbe_->bookProfile(histo, histo, 82, -41., 41., 2100, -100., 2000., "s");
  sprintf  (histo, "emean_vs_ieta_EH" );
  emean_vs_ieta_EH = dbe_->bookProfile(histo, histo, 82, -41., 41., 2100, -100., 2000., "s");
  

  sprintf  (histo, "emean_vs_ieta_E1" );
  emean_vs_ieta_E1 = dbe_->bookProfile(histo, histo, 82, -41., 41., 2100, -100., 2000., "s");
  sprintf  (histo, "emean_vs_ieta_H1" );
  emean_vs_ieta_H1 = dbe_->bookProfile(histo, histo, 82, -41., 41., 2100, -100., 2000., "s");
  sprintf  (histo, "emean_vs_ieta_EH1" );
  emean_vs_ieta_EH1 = dbe_->bookProfile(histo, histo, 82, -41., 41., 2100, -100., 2000., "s");
  

  sprintf  (histo, "CaloTowersTask_map_energy_E" );
  mapEnergy_E = dbe_->book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
  sprintf  (histo, "CaloTowersTask_map_energy_H");
  mapEnergy_H = dbe_->book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
  sprintf  (histo, "CaloTowersTask_map_energy_EH" );
  mapEnergy_EH = dbe_->book2D(histo, histo, 82, -41., 41., 72, 0., 72.);

  sprintf  (histo, "CaloTowersTask_map_Nentries" );
  mapEnergy_N = dbe_->book2D(histo, histo, 82, -41., 41., 72, 0., 72.);


  sprintf  (histo, "CaloTowersTask_map_occupancy" );
  occupancy_map = dbe_->book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
  sprintf  (histo, "CaloTowersTask_occupancy_vs_ieta" );
  occupancy_vs_ieta = dbe_->book1D(histo, histo, 82, -41, 41);



  if( isub == 1 || isub == 0) {
    sprintf (histo, "CaloTowersTask_sum_of_energy_HCAL_vs_ECAL_HB") ;
    meEnergyHcalvsEcal_HB    = dbe_->book2D(histo, histo, 500, 0., 500., 500, 0., 500.);
    
    sprintf (histo, "CaloTowersTask_energy_OUTER_HB" ) ;
    meEnergyHO_HB    = dbe_->book1D(histo, histo, 440, -200, 2000);   
    
    sprintf (histo, "CaloTowersTask_energy_HCAL_HB" ) ;
    meEnergyHcal_HB    = dbe_->book1D(histo, histo, 440, -200, 2000);  
    
    sprintf (histo, "CaloTowersTask_energy_ECAL_HB" ) ;
    meEnergyEcal_HB    = dbe_->book1D(histo, histo, 440, -200, 2000); 
    
    sprintf (histo, "CaloTowersTask_number_of_fired_towers_HB" ) ;
    meNumFiredTowers_HB = dbe_->book1D(histo, histo, 400, 0, 2000); 
    
    sprintf (histo, "CaloTowersTask_energy_of_ECAL_component_of_tower_HB" ) ;
    meEnergyEcalTower_HB = dbe_->book1D(histo, histo, 440, -200, 2000); 
    
    sprintf (histo, "CaloTowersTask_energy_of_HCAL_component_of_tower_HB" ) ;
    meEnergyHcalTower_HB = dbe_->book1D(histo, histo, 440 , -200 , 2000); 
    
    sprintf  (histo, "CaloTowersTask_energy_HcalPlusEcalPlusHO_HB" ) ;
    meTotEnergy_HB = dbe_->book1D(histo, histo,400, 0., 2000.) ;
    
    sprintf  (histo, "CaloTowersTask_map_energy_HB" );
    mapEnergy_HB = dbe_->book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
    sprintf  (histo, "CaloTowersTask_map_energy_HCAL_HB");
    mapEnergyHcal_HB = dbe_->book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
    sprintf  (histo, "CaloTowersTask_map_energy_ECAL_HB" );
    mapEnergyEcal_HB = dbe_->book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
    
    sprintf  (histo, "CaloTowersTask_MET_HB" ) ;
    MET_HB = dbe_->book1D(histo, histo, 500, 0. , 1000. ) ;
    
    sprintf  (histo, "CaloTowersTask_SET_HB" ) ;
    SET_HB = dbe_->book1D(histo, histo, 500, 0. , 5000. ) ;
    
    sprintf  (histo, "CaloTowersTask_phi_MET_HB" ) ;
    phiMET_HB = dbe_->book1D(histo, histo, 72, -3.1415926535898, 3.1415926535898 ) ;
  }

  if( isub == 2 || isub == 0) {
    sprintf (histo, "CaloTowersTask_sum_of_energy_HCAL_vs_ECAL_HE") ;
    meEnergyHcalvsEcal_HE    = dbe_->book2D(histo, histo, 500, 0., 500., 500, 0., 500.);
    
    sprintf (histo, "CaloTowersTask_energy_OUTER_HE" ) ;
    meEnergyHO_HE    = dbe_->book1D(histo, histo, 440, -200, 2000);   
    
    sprintf (histo, "CaloTowersTask_energy_HCAL_HE" ) ;
    meEnergyHcal_HE    = dbe_->book1D(histo, histo, 440, -200, 2000);  
    
    sprintf (histo, "CaloTowersTask_energy_ECAL_HE" ) ;
    meEnergyEcal_HE    = dbe_->book1D(histo, histo, 440, -200, 2000); 
    
    sprintf (histo, "CaloTowersTask_number_of_fired_towers_HE" ) ;
    meNumFiredTowers_HE = dbe_->book1D(histo, histo, 400, 0, 2000); 
    
    sprintf (histo, "CaloTowersTask_energy_of_ECAL_component_of_tower_HE" ) ;
    meEnergyEcalTower_HE = dbe_->book1D(histo, histo, 440, -200, 2000); 
    
    sprintf (histo, "CaloTowersTask_energy_of_HCAL_component_of_tower_HE" ) ;
    meEnergyHcalTower_HE = dbe_->book1D(histo, histo, 440 , -200 , 2000); 
    
    sprintf  (histo, "CaloTowersTask_energy_HcalPlusEcalPlusHO_HE" ) ;
    meTotEnergy_HE = dbe_->book1D(histo, histo,400, 0., 2000.) ;
    
    sprintf  (histo, "CaloTowersTask_map_energy_HE" );
    mapEnergy_HE = dbe_->book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
    sprintf  (histo, "CaloTowersTask_map_energy_HCAL_HE");
    mapEnergyHcal_HE = dbe_->book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
    sprintf  (histo, "CaloTowersTask_map_energy_ECAL_HE" );
    mapEnergyEcal_HE = dbe_->book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
    
    sprintf  (histo, "CaloTowersTask_MET_HE" ) ;
    MET_HE = dbe_->book1D(histo, histo, 500, 0. , 1000. ) ;
    
    sprintf  (histo, "CaloTowersTask_SET_HE" ) ;
    SET_HE = dbe_->book1D(histo, histo, 500, 0. , 5000. ) ;
    
    sprintf  (histo, "CaloTowersTask_phi_MET_HE" ) ;
    phiMET_HE = dbe_->book1D(histo, histo, 72, -3.1415926535898, 3.1415926535898 ) ;
  }


  if( isub == 3 || isub == 0) {
    sprintf (histo, "CaloTowersTask_sum_of_energy_HCAL_vs_ECAL_HF") ;
    meEnergyHcalvsEcal_HF    = dbe_->book2D(histo, histo, 500, 0., 500., 500, 0., 500.);
    
    sprintf (histo, "CaloTowersTask_energy_OUTER_HF" ) ;
    meEnergyHO_HF    = dbe_->book1D(histo, histo, 440, -200, 2000);   
    
    sprintf (histo, "CaloTowersTask_energy_HCAL_HF" ) ;
    meEnergyHcal_HF    = dbe_->book1D(histo, histo, 440, -200, 2000);  
    
    sprintf (histo, "CaloTowersTask_energy_ECAL_HF" ) ;
    meEnergyEcal_HF    = dbe_->book1D(histo, histo, 440, -200, 2000); 
    
    sprintf (histo, "CaloTowersTask_number_of_fired_towers_HF" ) ;
    meNumFiredTowers_HF = dbe_->book1D(histo, histo, 400, 0, 2000); 
    
    sprintf (histo, "CaloTowersTask_energy_of_ECAL_component_of_tower_HF" ) ;
    meEnergyEcalTower_HF = dbe_->book1D(histo, histo, 440, -200, 2000); 
    
    sprintf (histo, "CaloTowersTask_energy_of_HCAL_component_of_tower_HF" ) ;
    meEnergyHcalTower_HF = dbe_->book1D(histo, histo, 440 , -200 , 2000); 
    
    sprintf  (histo, "CaloTowersTask_energy_HcalPlusEcalPlusHO_HF" ) ;
    meTotEnergy_HF = dbe_->book1D(histo, histo,400, 0., 2000.) ;
    
    sprintf  (histo, "CaloTowersTask_map_energy_HF" );
    mapEnergy_HF = dbe_->book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
    sprintf  (histo, "CaloTowersTask_map_energy_HCAL_HF");
    mapEnergyHcal_HF = dbe_->book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
    sprintf  (histo, "CaloTowersTask_map_energy_ECAL_HF" );
    mapEnergyEcal_HF = dbe_->book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
    
    sprintf  (histo, "CaloTowersTask_MET_HF" ) ;
    MET_HF = dbe_->book1D(histo, histo, 500, 0. , 500. ) ;
    
    sprintf  (histo, "CaloTowersTask_SET_HF" ) ;
    SET_HF = dbe_->book1D(histo, histo, 500, 0. , 5000. ) ;
    
    sprintf  (histo, "CaloTowersTask_phi_MET_HF" ) ;
    phiMET_HF = dbe_->book1D(histo, histo, 72, -3.1415926535898, 3.1415926535898 ) ;

  }

}


CaloTowersValidation::~CaloTowersValidation() {

  // mean number of towers per ieta
  int nx = Ntowers_vs_ieta->getNbinsX();
  float cont;
  float fev = float(nevent);

  for (int i = 1; i <= nx; i++) {
    cont = Ntowers_vs_ieta -> getBinContent(i) / fev ;
    Ntowers_vs_ieta -> setBinContent(i,cont);
  }

  // mean energies & occupancies evaluation
  
  nx = mapEnergy_N->getNbinsX();    
  int ny = mapEnergy_N->getNbinsY();
  float cnorm;
  float phi_factor;    

  for (int i = 1; i <= nx; i++) {
    float sumphi = 0.;

    for (int j = 1; j <= ny; j++) {      

      // Emean
      cnorm   = mapEnergy_N -> getBinContent(i,j);
      if(cnorm > 0.000001) {
	
	cont = mapEnergy_E -> getBinContent(i,j) / cnorm ;
	mapEnergy_E -> setBinContent(i,j,cont);	      
	
	cont = mapEnergy_H -> getBinContent(i,j) / cnorm ;
	mapEnergy_H -> setBinContent(i,j,cont);	      
	
	cont = mapEnergy_EH -> getBinContent(i,j) / cnorm ;
	mapEnergy_EH -> setBinContent(i,j,cont);	      
      }
      
      // Occupancy
      cnorm   = occupancy_map -> getBinContent(i,j) / fev; 
      if(cnorm > 1.e-30) occupancy_map -> setBinContent(i,j,cnorm);

      sumphi += cnorm;

    } // end of iphy cycle (j)

    // phi-factor evaluation for occupancy_vs_ieta calculation
    int ieta = i - 42;        // -41 -1, 0 40 
    if(ieta >=0 ) ieta +=1;   // -41 -1, 1 41  - to make it detector-like
    
    if(ieta >= -20 && ieta <= 20 )
      {phi_factor = 72.;}
      else {
	if(ieta >= 40 || ieta <= -40 ) {phi_factor = 18.;}
        else 
	  phi_factor = 36.;
      }  
    if(ieta >= 0) ieta -= 1; // -41 -1, 0 40  - to bring back to histo num
    
    cnorm = sumphi / phi_factor;
    occupancy_vs_ieta->Fill(double(ieta), cnorm);


  } // end of ieta cycle (i)

  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
  
}

void CaloTowersValidation::endJob() { }

void CaloTowersValidation::beginJob(){ nevent = 0; }

void CaloTowersValidation::analyze(edm::Event const& event, edm::EventSetup const& c) {

  nevent++;

  bool     MC = false;
  double   phi_MC = 9999.;
  double   eta_MC = 9999.;


  edm::Handle<edm::HepMCProduct> evtMC;
  //  ev.getByLabel("VtxSmeared",evtMC);
  event.getByLabel("generator",evtMC);  // generator in late 310_preX
  if (!evtMC.isValid()) {
    std::cout << "no HepMCProduct found" << std::endl;    
  } else {
    MC=true;
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
    //    phi_MC = phip;
    //    eta_MC = etap;
    double pt  = (*p)->momentum().perp();
    if(pt > maxPt) { npart++; maxPt = pt; phi_MC = phip; eta_MC = etap; }
  }
  //  std::cout << "*** Max pT = " << maxPt <<  std::endl;  

  edm::Handle<CaloTowerCollection> towers;
  event.getByLabel(theCaloTowerCollectionLabel,towers);
  CaloTowerCollection::const_iterator cal;

  double met;
  double phimet;

  // ieta scan 
  double partR  = 0.3;
  double Rmin   = 9999.;
  double Econe  = 0.;
  double Hcone  = 0.;
  double Ee1    = 0.;
  double Eh1    = 0.;
  double ieta_MC = 9999;
  double iphi_MC = 9999;
  //  double  etaM   = 9999.;


  // HB   
  double sumEnergyHcal_HB = 0.;
  double sumEnergyEcal_HB = 0.;
  double sumEnergyHO_HB   = 0.;
  Int_t numFiredTowers_HB = 0;
  double metx_HB   =  0.;
  double mety_HB   =  0.;
  double metz_HB   =  0.;
  double sEt_HB    =  0.;
  // HE   
  double sumEnergyHcal_HE = 0.;
  double sumEnergyEcal_HE = 0.;
  double sumEnergyHO_HE   = 0.;
  Int_t numFiredTowers_HE = 0;
  double metx_HE   =  0.;
  double mety_HE   =  0.;
  double metz_HE   =  0.;
  double sEt_HE    =  0.;
  // HF   
  double sumEnergyHcal_HF = 0.;
  double sumEnergyEcal_HF = 0.;
  double sumEnergyHO_HF   = 0.;
  Int_t numFiredTowers_HF = 0;
  double metx_HF   =  0.;
  double mety_HF   =  0.;
  double metz_HF   =  0.;
  double sEt_HF    =  0.;

  for ( cal = towers->begin(); cal != towers->end(); ++cal ) {
    
    double eE   = cal->emEnergy();
    double eH   = cal->hadEnergy();
    double eHO  = cal->outerEnergy();
    double etaT = cal->eta();
    double phiT = cal->phi();
    double en   = cal->energy();

    math::RhoEtaPhiVector mom(cal->et(), cal->eta(), cal->phi());
			      //  Vector mom  = cal->momentum(); 
  
    // cell properties    
    CaloTowerDetId idT = cal->id();
    int ieta = idT.ieta();
    if(ieta > 0) ieta -= 1;
    int iphi = idT.iphi();


    double r    = dR(eta_MC, phi_MC, etaT, phiT);

    if( r < partR ){
      Econe += eE; 
      Hcone += eH; 

      // closest to MC
      if(r < Rmin) { 
        if( fabs(eta_MC) < 3.0 && (ieta > 29 || ieta < -29)) {;}
	else {    
	  Rmin = r;
	  ieta_MC = ieta; 
	  iphi_MC = iphi; 
	  Ee1     = eE;
	  Eh1     = eH;
	}
      }
    }
      

    Ntowers_vs_ieta -> Fill(double(ieta),1.);
    occupancy_map -> Fill(double(ieta),double(iphi));
    

    if((isub == 0 || isub == 1) 
       && (fabs(etaT) <  etaMax[0] && fabs(etaT) >= etaMin[0] )) {
      mapEnergy_HB     -> Fill(double(ieta), double(iphi), en); 
      mapEnergyHcal_HB -> Fill(double(ieta), double(iphi), eH); 
      mapEnergyEcal_HB -> Fill(double(ieta), double(iphi), eE); 
      
      //      std::cout << " e_ecal = " << eE << std::endl;
      
      //  simple sums
      sumEnergyHcal_HB += eH;
      sumEnergyEcal_HB += eE;
      sumEnergyHO_HB   += eHO;
      
      numFiredTowers_HB++;
      
      meEnergyEcalTower_HB->Fill(eE);
      meEnergyHcalTower_HB->Fill(eH);    
      
      // MET, SET & phimet
      //  double  etT = cal->et();
      metx_HB += mom.x();   
      mety_HB += mom.y();  //etT * sin(phiT);          
      sEt_HB  += en;    
    }
   
    if((isub == 0 || isub == 2) 
       && (fabs(etaT) <  etaMax[1] && fabs(etaT) >= etaMin[1] )) {
      mapEnergy_HE     -> Fill(double(ieta), double(iphi), en); 
      mapEnergyHcal_HE -> Fill(double(ieta), double(iphi), eH); 
      mapEnergyEcal_HE -> Fill(double(ieta), double(iphi), eE); 
      
      //      std::cout << " e_ecal = " << eE << std::endl;
      
      //  simple sums
      sumEnergyHcal_HE += eH;
      sumEnergyEcal_HE += eE;
      sumEnergyHO_HE   += eHO;
      
      numFiredTowers_HE++;
      
      meEnergyEcalTower_HE->Fill(eE);
      meEnergyHcalTower_HE->Fill(eH);    
      
      // MET, SET & phimet
      //  double  etT = cal->et();
      metx_HE += mom.x();   
      mety_HE += mom.y();  //etT * sin(phiT);          
      sEt_HE  += en;    
    }

    if((isub == 0 || isub == 3) 
       && (fabs(etaT) <  etaMax[2] && fabs(etaT) >= etaMin[2] )) {
      mapEnergy_HF     -> Fill(double(ieta), double(iphi), en); 
      mapEnergyHcal_HF -> Fill(double(ieta), double(iphi), eH); 
      mapEnergyEcal_HF -> Fill(double(ieta), double(iphi), eE); 
      
      //      std::cout << " e_ecal = " << eE << std::endl;
      
      //  simple sums
      sumEnergyHcal_HF += eH;
      sumEnergyEcal_HF += eE;
      sumEnergyHO_HF   += eHO;
      
      numFiredTowers_HF++;
      
      meEnergyEcalTower_HF->Fill(eE);
      meEnergyHcalTower_HF->Fill(eH);    
      
      // MET, SET & phimet
      //  double  etT = cal->et();
      metx_HF += mom.x();   
      mety_HF += mom.y();  //etT * sin(phiT);          
      sEt_HF  += en;    
    }



  } // end of Towers cycle 

  emean_vs_ieta_E  -> Fill(double(ieta_MC), Econe); 
  emean_vs_ieta_H  -> Fill(double(ieta_MC), Hcone); 
  emean_vs_ieta_EH -> Fill(double(ieta_MC), Econe+Hcone); 
  
  emean_vs_ieta_E1  -> Fill(double(ieta_MC), Ee1); 
  emean_vs_ieta_H1  -> Fill(double(ieta_MC), Eh1); 
  emean_vs_ieta_EH1 -> Fill(double(ieta_MC), Ee1+Eh1); 

  mapEnergy_E -> Fill(double(ieta_MC), double(iphi_MC), Ee1); 
  mapEnergy_H -> Fill(double(ieta_MC), double(iphi_MC), Eh1); 
  mapEnergy_EH -> Fill(double(ieta_MC), double(iphi_MC), Ee1+Eh1); 
  mapEnergy_N  -> Fill(double(ieta_MC), double(iphi_MC), 1.); 


  if(isub == 0 || isub == 1) {
    met    = sqrt(metx_HB*metx_HB + mety_HB*mety_HB);
    Vector metv(metx_HB,mety_HB,metz_HB);
    phimet = metv.phi();
    
    meEnergyHcalvsEcal_HB->Fill(sumEnergyEcal_HB, sumEnergyHcal_HB);
    meEnergyHO_HB->        Fill(sumEnergyHO_HB);
    meEnergyHcal_HB->      Fill(sumEnergyHcal_HB);
    meEnergyEcal_HB->      Fill(sumEnergyEcal_HB);
    meNumFiredTowers_HB->  Fill(numFiredTowers_HB);
    meTotEnergy_HB->       Fill(sumEnergyHcal_HB+sumEnergyEcal_HB
				+sumEnergyHO_HB);    
    MET_HB    -> Fill (met); 
    phiMET_HB -> Fill (phimet); 
    SET_HB    -> Fill (sEt_HB); 
  }    


  if(isub == 0 || isub == 2) {
    met    = sqrt(metx_HE*metx_HE + mety_HE*mety_HE);
    Vector metv(metx_HE,mety_HE,metz_HE);
    phimet = metv.phi();
    
    meEnergyHcalvsEcal_HE->Fill(sumEnergyEcal_HE, sumEnergyHcal_HE);
    meEnergyHO_HE->        Fill(sumEnergyHO_HE);
    meEnergyHcal_HE->      Fill(sumEnergyHcal_HE);
    meEnergyEcal_HE->      Fill(sumEnergyEcal_HE);
    meNumFiredTowers_HE->  Fill(numFiredTowers_HE);
    meTotEnergy_HE->       Fill(sumEnergyHcal_HE+sumEnergyEcal_HE
				+sumEnergyHO_HE);    
    MET_HE    -> Fill (met); 
    phiMET_HE -> Fill (phimet); 
    SET_HE    -> Fill (sEt_HE); 
  }

  if(isub == 0 || isub == 3) {
    met    = sqrt(metx_HF*metx_HF + mety_HF*mety_HF);
    Vector metv(metx_HF,mety_HF,metz_HF);
    phimet = metv.phi();
    
    meEnergyHcalvsEcal_HF->Fill(sumEnergyEcal_HF, sumEnergyHcal_HF);
    meEnergyHO_HF->        Fill(sumEnergyHO_HF);
    meEnergyHcal_HF->      Fill(sumEnergyHcal_HF);
    meEnergyEcal_HF->      Fill(sumEnergyEcal_HF);
    meNumFiredTowers_HF->  Fill(numFiredTowers_HF);
    meTotEnergy_HF->       Fill(sumEnergyHcal_HF+sumEnergyEcal_HF
				+sumEnergyHO_HF);    
    MET_HF    -> Fill (met); 
    phiMET_HF -> Fill (phimet); 
    SET_HF    -> Fill (sEt_HF); 
  }

}

double CaloTowersValidation::dR(double eta1, double phi1, double eta2, double phi2) { 
  double PI = 3.1415926535898;
  double deltaphi= phi1 - phi2;
  if( phi2 > phi1 ) { deltaphi= phi2 - phi1;}
  if(deltaphi > PI) { deltaphi = 2.*PI - deltaphi;}
  double deltaeta = eta2 - eta1;
  double tmp = sqrt(deltaeta* deltaeta + deltaphi*deltaphi);
  return tmp;
}

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CaloTowersValidation);

