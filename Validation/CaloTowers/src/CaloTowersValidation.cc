#include "Validation/CaloTowers/interface/CaloTowersValidation.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

CaloTowersValidation::CaloTowersValidation(edm::ParameterSet const& conf):
  theCaloTowerCollectionLabel(conf.getUntrackedParameter<std::string>("CaloTowerCollectionLabel"))
{
  // DQM ROOT output
  outputFile_ = conf.getUntrackedParameter<std::string>("outputFile", "myfile.root");

  
  if ( outputFile_.size() != 0 ) {
    edm::LogInfo("OutputInfo") << " Hcal RecHit Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("OutputInfo") << " Hcal RecHit Task histograms will NOT be saved";
  }
  
  dbe_ = 0;

  // get hold of back-end interface
  dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();
   

  Char_t histo[20];

 

  if ( dbe_ ) {
    std::cout << " dbe_->setCurrentFolder" << std::endl; 
    dbe_->setCurrentFolder("CaloTowersTask");
  
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
    
  }
}


CaloTowersValidation::~CaloTowersValidation() {
   
  std::cout << " outputFile_.size() =  " << outputFile_.size() << std::endl;
  std::cout << " dbe_ = " << dbe_ << std::endl;
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
  
}

void CaloTowersValidation::endJob() {
  
  std::cout << " outputFile_.size() =  " << outputFile_.size() << std::endl;
  std::cout << " dbe_ = " << dbe_ << std::endl; 
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
  
}
void CaloTowersValidation::beginJob(const edm::EventSetup& c){

}
void CaloTowersValidation::analyze(edm::Event const& event, edm::EventSetup const& c) {

  edm::Handle<CaloTowerCollection> towers;
  event.getByLabel(theCaloTowerCollectionLabel,towers);
  CaloTowerCollection::const_iterator cal;

  // sum of et for HCAL part of tower   
  Double_t sumEnergyHcal =0.;
  // sum of et for ECAL part of tower  
  Double_t sumEnergyEcal =0.;
  //
  Double_t sumEnergyHO =0.;
  //
  Int_t numFiredTowers = 0;

 for ( cal = towers->begin(); cal != towers->end(); ++cal ) {

      double eE = cal->emEnergy();
//      std::cout << " e_ecal = " << eE << std::endl;
      double eH = cal->hadEnergy();
//      std::cout << " e_hcal = " << eH << std::endl;
      double eHO = cal->outerEnergy();
//      std::cout << " e_HO = " << eHO << std::endl;
      sumEnergyHcal += eH;
      sumEnergyEcal += eE;
      sumEnergyHO += eHO;
      
      numFiredTowers++;


      meEnergyEcalTower->Fill(eE);
      meEnergyHcalTower->Fill(eH);    
 }

 meEnergyHcalvsEcal->Fill(sumEnergyEcal, sumEnergyHcal);
 meEnergyHO -> Fill(sumEnergyHO);
 meEnergyHcal-> Fill(sumEnergyHcal);
 meEnergyEcal-> Fill(sumEnergyEcal);
 meNumFiredTowers->Fill(numFiredTowers);
 meTotEnergy->Fill(sumEnergyHcal+sumEnergyEcal+sumEnergyHO);
}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CaloTowersValidation);

