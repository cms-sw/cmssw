/*
 * \file EcalEndcapRecHitsValidation.cc
 *
 * $Date: 2006/05/22 $
 * \author C. Rovelli
 *
 */

#include <Validation/EcalRecHits/interface/EcalEndcapRecHitsValidation.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>

EcalEndcapRecHitsValidation::EcalEndcapRecHitsValidation(const ParameterSet& ps){

  // ---------------------- 
  uncalibrecHitProducer_     = ps.getParameter<std::string>("uncalibrecHitProducer");
  EEuncalibrechitCollection_ = ps.getParameter<std::string>("EEuncalibrechitCollection");
  
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
  meEEUncalibRecHitsOccupancy_     = 0;
  meEEUncalibRecHitsAmplitude_     = 0;
  meEEUncalibRecHitsPedestal_      = 0;
  meEEUncalibRecHitsJitter_        = 0;
  meEEUncalibRecHitsChi2_          = 0;

  // ---------------------- 
  Char_t histo[200];
   
  if ( dbe_ ) 
    {
      dbe_->setCurrentFolder("EcalEndcapRecHitsTask");
      
      sprintf (histo, "EE Occupancy" );  
      meEEUncalibRecHitsOccupancy_ = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
      
      sprintf (histo, "EE Amplitude" );
      meEEUncalibRecHitsAmplitude_ = dbe_->book1D(histo, histo, 100, 0., 1000.);
      
      sprintf (histo, "EE Pedestal" );
      meEEUncalibRecHitsPedestal_ = dbe_->book1D(histo, histo, 100, 0., 1000.);
      
      sprintf (histo, "EE Jitter" );
      meEEUncalibRecHitsJitter_ = dbe_->book1D(histo, histo, 100, -1000., 1000.);
      
      sprintf (histo, "EE Chi2" );
      meEEUncalibRecHitsChi2_ = dbe_->book1D(histo, histo, 100, 0., 100.);
    }
}

EcalEndcapRecHitsValidation::~EcalEndcapRecHitsValidation(){   

}

void EcalEndcapRecHitsValidation::beginJob(const EventSetup& c){  

}

void EcalEndcapRecHitsValidation::endJob(){

}

void EcalEndcapRecHitsValidation::analyze(const Event& e, const EventSetup& c){

  Handle< EEUncalibratedRecHitCollection > EcalUncalibRecHitEE;

  try {
    e.getByLabel( uncalibrecHitProducer_, EEuncalibrechitCollection_, EcalUncalibRecHitEE);
  } catch ( std::exception& ex ) {
    edm::LogError("EcalRecHitsTaskError") << "Error! can't get the product " << EEuncalibrechitCollection_.c_str() << std::endl;
  }



  // ---------------------- 
  // loop over UncalibRecHits
  const EEUncalibratedRecHitCollection *EEUncalibRecHit = EcalUncalibRecHitEE.product() ;
  
  for (EcalUncalibratedRecHitCollection::const_iterator uncalibRecHit = EEUncalibRecHit->begin(); uncalibRecHit != EEUncalibRecHit->end() ; ++uncalibRecHit)
    {
      EEDetId EEid = EEDetId(uncalibRecHit->id());

      if (meEEUncalibRecHitsOccupancy_) meEEUncalibRecHitsOccupancy_ ->Fill(EEid.ix(), EEid.iy());  
      if (meEEUncalibRecHitsAmplitude_) meEEUncalibRecHitsAmplitude_ ->Fill(uncalibRecHit->amplitude());
      if (meEEUncalibRecHitsPedestal_)  meEEUncalibRecHitsPedestal_  ->Fill(uncalibRecHit->pedestal());
      if (meEEUncalibRecHitsJitter_)    meEEUncalibRecHitsJitter_    ->Fill(uncalibRecHit->jitter());
      if (meEEUncalibRecHitsChi2_)      meEEUncalibRecHitsChi2_      ->Fill(uncalibRecHit->chi2());
      
    }  // loop over the UncalibratedRecHitCollection

}
