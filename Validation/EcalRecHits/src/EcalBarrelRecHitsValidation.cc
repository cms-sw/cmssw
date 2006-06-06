/*
 * \file EcalBarrelRecHitsValidation.cc
 *
 * $Date: 2006/05/22 $
 * \author C. Rovelli
 *
 */

#include <Validation/EcalRecHits/interface/EcalBarrelRecHitsValidation.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>

EcalBarrelRecHitsValidation::EcalBarrelRecHitsValidation(const ParameterSet& ps){

  // ---------------------- 
  uncalibrecHitProducer_     = ps.getParameter<std::string>("uncalibrecHitProducer");
  EBuncalibrechitCollection_ = ps.getParameter<std::string>("EBuncalibrechitCollection");
  
  // ---------------------- 
  // verbosity switch 
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);
  
  if ( verbose_ ) {
    cout << " verbose switch is ON" << endl;
  } else {
    cout << " verbose switch is OFF" << endl;
  }
  
  // ----------------------                 
  // get hold of back-end interface 
  dbe_ = 0;
  dbe_ = Service<DaqMonitorBEInterface>().operator->();                   
  if ( dbe_ ) {
    if ( verbose_ ) {
      dbe_->setVerbose(1);
    } else {
      dbe_->setVerbose(0);
    }
  }                                                                  
  if ( dbe_ ) {
    if ( verbose_ ) dbe_->showDirStructure();
  }


  // ----------------------   
  meEBUncalibRecHitsOccupancy_     = 0;
  meEBUncalibRecHitsAmplitude_     = 0;
  meEBUncalibRecHitsPedestal_      = 0;
  meEBUncalibRecHitsJitter_        = 0;
  meEBUncalibRecHitsChi2_          = 0;
  for (int i=0; i<36 ; i++) 
    {
      meEBUncalibRecHitAmplMap_[i] = 0;
      meEBUncalibRecHitPedMap_[i]  = 0;
    }

  // ---------------------- 
  Char_t histo[200];
   
  if ( dbe_ ) 
    {
      dbe_->setCurrentFolder("EcalBarrelRecHitsTask");
      
      sprintf (histo, "EB Occupancy" );  
      meEBUncalibRecHitsOccupancy_ = dbe_->book2D(histo, histo, 170, -85., 85., 360, 0., 360.);
      
      sprintf (histo, "EB Amplitude" );
      meEBUncalibRecHitsAmplitude_ = dbe_->book1D(histo, histo, 100, 0., 1000.);
      
      sprintf (histo, "EB Pedestal" );
      meEBUncalibRecHitsPedestal_ = dbe_->book1D(histo, histo, 100, 0., 1000.);
      
      sprintf (histo, "EB Jitter" );
      meEBUncalibRecHitsJitter_ = dbe_->book1D(histo, histo, 100, -1000., 1000.);
      
      sprintf (histo, "EB Chi2" );
      meEBUncalibRecHitsChi2_ = dbe_->book1D(histo, histo, 100, 0., 100.);
      
      for (int i=0; i<36 ; i++) 
	{
	  sprintf(histo, "EB Amp SM%02d", i+1);
	  meEBUncalibRecHitAmplMap_[i] = dbe_->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20.,100,0.,1000.);
	  
	  sprintf(histo, "EB Ped SM%02d", i+1);
	  meEBUncalibRecHitPedMap_[i] = dbe_->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20.,100,0.,1000.);
	}
    }
}

EcalBarrelRecHitsValidation::~EcalBarrelRecHitsValidation(){   

}

void EcalBarrelRecHitsValidation::beginJob(const EventSetup& c){  

}

void EcalBarrelRecHitsValidation::endJob(){

}

void EcalBarrelRecHitsValidation::analyze(const Event& e, const EventSetup& c){
  
  Handle< EBUncalibratedRecHitCollection > EcalUncalibRecHitEB;
  try {
    e.getByLabel( uncalibrecHitProducer_, EBuncalibrechitCollection_, EcalUncalibRecHitEB);
  } catch ( std::exception& ex ) {
    edm::LogError("EcalRecHitsTaskError") << "Error! can't get the product " << EBuncalibrechitCollection_.c_str() << std::endl;
  }




  // ---------------------- 
  // loop over UncalibRecHits
  const EBUncalibratedRecHitCollection *EBUncalibRecHit = EcalUncalibRecHitEB.product() ;
  
  for (EcalUncalibratedRecHitCollection::const_iterator uncalibRecHit = EBUncalibRecHit->begin(); uncalibRecHit != EBUncalibRecHit->end() ; ++uncalibRecHit)
    {
      EBDetId EBid = EBDetId(uncalibRecHit->id());

      if (meEBUncalibRecHitsOccupancy_) meEBUncalibRecHitsOccupancy_ ->Fill(EBid.ieta(), EBid.iphi());
      if (meEBUncalibRecHitsAmplitude_) meEBUncalibRecHitsAmplitude_ ->Fill(uncalibRecHit->amplitude());
      if (meEBUncalibRecHitsPedestal_)  meEBUncalibRecHitsPedestal_  ->Fill(uncalibRecHit->pedestal());
      if (meEBUncalibRecHitsJitter_)    meEBUncalibRecHitsJitter_    ->Fill(uncalibRecHit->jitter());
      if (meEBUncalibRecHitsChi2_)      meEBUncalibRecHitsChi2_      ->Fill(uncalibRecHit->chi2());

      int ic = EBid.ic();              
      int ie = (ic-1)/20 + 1;          
      int ip = (ic-1)%20 + 1;
      int ism = EBid.ism();
      float xie = ie - 0.5;
      float xip = ip - 0.5;

      LogDebug("EcalRecHitsTask") << " det id = "     << EBid;
      LogDebug("EcalRecHitsTask") << " sm, eta, phi " << ism << " " << ie << " " << ip << endl;

      
      meEBUncalibRecHitPedMap_[ism-1] ->Fill(xie, xip, uncalibRecHit->pedestal());
      meEBUncalibRecHitAmplMap_[ism-1]->Fill(xie, xip, uncalibRecHit->amplitude());
      
    }  // loop over the UncalibratedRecHitCollection

}
