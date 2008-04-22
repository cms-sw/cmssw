#include "Validation/CaloTowers/interface/CaloTowersValidation.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

CaloTowersValidation::CaloTowersValidation(edm::ParameterSet const& conf):
  theCaloTowerCollectionLabel(conf.getUntrackedParameter<std::string>("CaloTowerCollectionLabel"))
{
  // DQM ROOT output
  outputFile_ = conf.getUntrackedParameter<std::string>("outputFile", "myfile.root");

  hcalselector_ = conf.getUntrackedParameter<std::string>("hcalselector", "all");

  etaMin = 0.;
  etaMax = 2.95;
  if(hcalselector_ == "HF" || hcalselector_ == "all") {
    etaMin = 2.93;
    etaMax = 5.2;
  } 
  if(hcalselector_ == "HB") {
    etaMin = 0.;
    etaMax = 1.3;
  }
  if(hcalselector_ == "HE") {
    etaMin = 1.48;
    etaMax = 2.93;
  }
  
  if ( outputFile_.size() != 0 ) {
    edm::LogInfo("OutputInfo") << " Hcal RecHit Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("OutputInfo") << " Hcal RecHit Task histograms will NOT be saved";
  }
  
  dbe_ = 0;

  // get hold of back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();
   
  Char_t histo[20];

  if ( dbe_ ) {
    std::cout << " dbe_->setCurrentFolder" << std::endl; 
    dbe_->setCurrentFolder("CaloTowersV/CaloTowersTask");
  
    sprintf (histo, "CaloTowersTask_sum_of_energy_HCAL_vs_ECAL" ) ;
    meEnergyHcalvsEcal    = dbe_->book2D(histo, histo, 220, -20 , 200., 220, -20., 200.);

    sprintf (histo, "CaloTowersTask_energy_OUTER" ) ;
    meEnergyHO    = dbe_->book1D(histo, histo, 15, 0 , 30);   

    sprintf (histo, "CaloTowersTask_energy_HCAL" ) ;
    meEnergyHcal    = dbe_->book1D(histo, histo, 75, 0 , 150);  

    sprintf (histo, "CaloTowersTask_energy_ECAL" ) ;
    meEnergyEcal    = dbe_->book1D(histo, histo, 75, 0 , 150); 

    sprintf (histo, "CaloTowersTask_number_of_fired_towers" ) ;
    meNumFiredTowers = dbe_->book1D(histo, histo, 75, 0 , 150); 

    sprintf (histo, "CaloTowersTask_energy_of_ECAL_component_of_tower" ) ;
    meEnergyEcalTower = dbe_->book1D(histo, histo, 240 , -20 , 100); 
  
    sprintf (histo, "CaloTowersTask_energy_of_HCAL_component_of_tower" ) ;
    meEnergyHcalTower = dbe_->book1D(histo, histo, 240 , -20 , 100); 

    sprintf  (histo, "CaloTowersTask_energy_HcalPlusEcalPlusHO" ) ;
    meTotEnergy = dbe_->book1D(histo, histo,100, 0. , 200. ) ;
    
    sprintf  (histo, "CaloTowersTask_map_energy" );
    mapEnergy = dbe_->book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
    sprintf  (histo, "CaloTowersTask_map_energy_HCAL");
    mapEnergyHcal = dbe_->book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
    sprintf  (histo, "CaloTowersTask_map_energy_ECAL" );
    mapEnergyEcal = dbe_->book2D(histo, histo, 82, -41., 41., 72, 0., 72.);

    sprintf  (histo, "CaloTowersTask_MET" ) ;
    MET = dbe_->book1D(histo, histo, 100, 0. , 200. ) ;
    
    sprintf  (histo, "CaloTowersTask_SET" ) ;
    SET = dbe_->book1D(histo, histo, 100, 0. , 200. ) ;
    
    sprintf  (histo, "CaloTowersTask_phi_MET" ) ;
    phiMET = dbe_->book1D(histo, histo, 72, -3.1415926535898, 3.1415926535898 ) ;
    /*
    sprintf (histo, "CaloTowersTask_profile_HCAL_cone_energy_vs eta");
    meRecHitSimHitProfileHF  = dbe_->bookProfile(histo, histo, 60, 0., 60., 250, 0., 500.);  
    
    */

  }
}


CaloTowersValidation::~CaloTowersValidation() {
   
  std::cout << " outputFile_.size() =  " << outputFile_.size() << std::endl;
  std::cout << " dbe_ = " << dbe_ << std::endl;
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
  
}

void CaloTowersValidation::endJob() { }

void CaloTowersValidation::beginJob(const edm::EventSetup& c){

}
void CaloTowersValidation::analyze(edm::Event const& event, edm::EventSetup const& c) {

  edm::Handle<CaloTowerCollection> towers;
  event.getByLabel(theCaloTowerCollectionLabel,towers);
  CaloTowerCollection::const_iterator cal;

  // energy sum of HCAL part of tower   
  double sumEnergyHcal = 0.;
  double sumEnergyEcal = 0.;
  double sumEnergyHO   = 0.;
  //
  Int_t numFiredTowers = 0;
  //
  double met;
  double metx   =  0.;
  double mety   =  0.;
  double metz   =  0.;
  double sEt    =  0.;
  double phimet =  0.;

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
    
    mapEnergy     -> Fill(double(ieta), double(iphi), en); 
    mapEnergyHcal -> Fill(double(ieta), double(iphi), eH); 
    mapEnergyEcal -> Fill(double(ieta), double(iphi), eE); 
    
    //      std::cout << " e_ecal = " << eE << std::endl;
    
    //  simple sums

    if(fabs(etaT) < etaMax && fabs(etaT) >= etaMin ) {
    
      sumEnergyHcal += eH;
      sumEnergyEcal += eE;
      sumEnergyHO   += eHO;
      
      numFiredTowers++;
      
      meEnergyEcalTower->Fill(eE);
      meEnergyHcalTower->Fill(eH);    
      
    /*    
00032   const Vector & momentum() const { return momentum_; }
00033   double et() const { return momentum_.rho(); }
00034   double energy() const { return momentum_.r(); }
00035   double eta() const { return momentum_.eta(); }
00036   double phi() const { return momentum_.phi(); }
00037   double emEnergy() const { return emEt_ * cosh( eta() ); }
00038   double hadEnergy() const { return hadEt_ * cosh( eta() ); }
00039   double outerEnergy() const { return outerEt_ * cosh( eta() ); }
    */

    // MET, SET & phimet
    //  double  etT = cal->et();
      metx += mom.x();   
      mety += mom.y();  //etT * sin(phiT);          
      sEt  += en;    
    }
  }
  
  met    = sqrt(metx*metx + mety*mety);
  Vector metv(metx,mety,metz);
  phimet = metv.phi();
    
  meEnergyHcalvsEcal->Fill(sumEnergyEcal, sumEnergyHcal);
  meEnergyHO -> Fill(sumEnergyHO);
  meEnergyHcal-> Fill(sumEnergyHcal);
  meEnergyEcal-> Fill(sumEnergyEcal);
  meNumFiredTowers->Fill(numFiredTowers);
  meTotEnergy->Fill(sumEnergyHcal+sumEnergyEcal+sumEnergyHO);

  MET    -> Fill (met); 
  SET    -> Fill (sEt); 
  phiMET -> Fill (phimet); 

}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMStore.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CaloTowersValidation);

