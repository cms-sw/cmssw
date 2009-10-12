#include "Validation/CaloTowers/interface/CaloTowersValidation.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

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
   
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
  
}

void CaloTowersValidation::endJob() { }

void CaloTowersValidation::beginJob(){

}
void CaloTowersValidation::analyze(edm::Event const& event, edm::EventSetup const& c) {

  edm::Handle<CaloTowerCollection> towers;
  event.getByLabel(theCaloTowerCollectionLabel,towers);
  CaloTowerCollection::const_iterator cal;

  double met;
  double phimet;

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
    //      double phiT = cal->eta();
    double en   = cal->energy();

    math::RhoEtaPhiVector mom(cal->et(), cal->eta(), cal->phi());
			      //  Vector mom  = cal->momentum(); 
  
    // cell properties    
    CaloTowerDetId idT = cal->id();
    int ieta = idT.ieta();
    int iphi = idT.iphi();

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


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CaloTowersValidation);

