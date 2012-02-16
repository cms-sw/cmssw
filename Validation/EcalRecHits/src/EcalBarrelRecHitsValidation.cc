/*
 * \file EcalBarrelRecHitsValidation.cc
 *
 * $Date: 2009/07/28 09:32:01 $
 * \author C. Rovelli
 *
 */

#include <Validation/EcalRecHits/interface/EcalBarrelRecHitsValidation.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include "DQMServices/Core/interface/DQMStore.h"

using namespace cms;
using namespace edm;
using namespace std;

EcalBarrelRecHitsValidation::EcalBarrelRecHitsValidation(const ParameterSet& ps){
  
  // ---------------------- 
  EBdigiCollection_          = ps.getParameter<edm::InputTag>("EBdigiCollection");
  EBuncalibrechitCollection_ = ps.getParameter<edm::InputTag>("EBuncalibrechitCollection");
  
  // ---------------------- 
  // verbosity switch 
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);
    
  // ----------------------                 
  // get hold of back-end interface 
  dbe_ = 0;
  dbe_ = Service<DQMStore>().operator->();                   
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
  meEBUncalibRecHitsOccupancy_             = 0;
  meEBUncalibRecHitsAmplitude_             = 0;
  meEBUncalibRecHitsPedestal_              = 0;
  meEBUncalibRecHitsJitter_                = 0;
  meEBUncalibRecHitsChi2_                  = 0;
  meEBUncalibRecHitMaxSampleRatio_         = 0;
  meEBUncalibRecHitsOccupancyGt100adc_     = 0;
  meEBUncalibRecHitsAmplitudeGt100adc_     = 0;
  meEBUncalibRecHitsPedestalGt100adc_      = 0;
  meEBUncalibRecHitsJitterGt100adc_        = 0;
  meEBUncalibRecHitsChi2Gt100adc_          = 0;
  meEBUncalibRecHitMaxSampleRatioGt100adc_ = 0;
  meEBUncalibRecHitsAmpFullMap_            = 0;
  meEBUncalibRecHitsPedFullMap_            = 0;
  for (int i=0; i<36 ; i++) 
    {
      meEBUncalibRecHitAmplMap_[i] = 0;
      meEBUncalibRecHitPedMap_[i]  = 0;
    }

  // ---------------------- 
  Char_t histo[200];
   
  if ( dbe_ ) 
    {
      dbe_->setCurrentFolder("EcalRecHitsV/EcalBarrelRecHitsTask");
      
      sprintf (histo, "EB Occupancy" );  
      meEBUncalibRecHitsOccupancy_ = dbe_->book2D(histo, histo, 170, -85., 85., 360, 0., 360.);
      
      sprintf (histo, "EB Amplitude" );
      meEBUncalibRecHitsAmplitude_ = dbe_->book1D(histo, histo, 201, -20., 4000.);
      
      sprintf (histo, "EB Pedestal" );
      meEBUncalibRecHitsPedestal_ = dbe_->book1D(histo, histo, 50, 190., 210.);
      
      sprintf (histo, "EB Jitter" );
      meEBUncalibRecHitsJitter_ = dbe_->book1D(histo, histo, 100, 0., 100.);
      
      sprintf (histo, "EB Chi2" );
      meEBUncalibRecHitsChi2_ = dbe_->book1D(histo, histo, 100, 18000., 22000.);

      sprintf (histo, "EB RecHit Max Sample Ratio"); 
      meEBUncalibRecHitMaxSampleRatio_ = dbe_->book1D(histo, histo, 120, 0.90, 1.05);

      sprintf (histo, "EB Occupancy gt 100 adc counts" );
      meEBUncalibRecHitsOccupancyGt100adc_ = dbe_->book2D(histo, histo, 170, -85., 85., 360, 0., 360.);
      
      sprintf (histo, "EB Amplitude gt 100 adc counts" );
      meEBUncalibRecHitsAmplitudeGt100adc_ = dbe_->book1D(histo, histo, 200, 0., 4000.);

      sprintf (histo, "EB Pedestal gt 100 adc counts" );
      meEBUncalibRecHitsPedestalGt100adc_ = dbe_->book1D(histo, histo, 50, 190., 210.);

      sprintf (histo, "EB Jitter gt 100 adc counts" );
      meEBUncalibRecHitsJitterGt100adc_ = dbe_->book1D(histo, histo, 100, 0., 100.);

      sprintf (histo, "EB Chi2 gt 100 adc counts" );
      meEBUncalibRecHitsChi2Gt100adc_ = dbe_->book1D(histo, histo, 100, 18000., 22000.);
    
      sprintf (histo, "EB RecHit Max Sample Ratio gt 100 adc counts"); 
      meEBUncalibRecHitMaxSampleRatioGt100adc_ = dbe_->book1D(histo, histo, 120, 0.90, 1.05);
      
      sprintf (histo, "EB Amplitude Full Map");
      meEBUncalibRecHitsAmpFullMap_ = dbe_->bookProfile2D(histo, histo, 170, -85., 85., 360, 0., 360., 200, 0., 4000.);

      sprintf (histo, "EB Pedestal Full Map");
      meEBUncalibRecHitsPedFullMap_ = dbe_->bookProfile2D(histo, histo, 170, -85., 85., 360, 0., 360., 50, 194., 201.);

      for (int i=0; i<36 ; i++) 
	{
	  sprintf(histo, "EB Amp SM%02d", i+1);
	  meEBUncalibRecHitAmplMap_[i] = dbe_->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 200, 0., 4000.);
	  
	  sprintf(histo, "EB Ped SM%02d", i+1);
	  meEBUncalibRecHitPedMap_[i] = dbe_->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 50, 194., 201.);
	}
    }
}

EcalBarrelRecHitsValidation::~EcalBarrelRecHitsValidation(){   

}

void EcalBarrelRecHitsValidation::beginJob(){  

}

void EcalBarrelRecHitsValidation::endJob(){

}

void EcalBarrelRecHitsValidation::analyze(const Event& e, const EventSetup& c){

  
  const EBUncalibratedRecHitCollection *EBUncalibRecHit = 0;
  Handle< EBUncalibratedRecHitCollection > EcalUncalibRecHitEB;
  e.getByLabel( EBuncalibrechitCollection_, EcalUncalibRecHitEB);
  if (EcalUncalibRecHitEB.isValid()) {
    EBUncalibRecHit = EcalUncalibRecHitEB.product();
  } else {
    return;
  }

  bool skipDigis = false;
  const EBDigiCollection *EBDigi = 0;
  Handle< EBDigiCollection > EcalDigiEB;
  e.getByLabel( EBdigiCollection_, EcalDigiEB);
  if (EcalDigiEB.isValid()) {
    EBDigi = EcalDigiEB.product();    
  } else {
    skipDigis = true;
  }

  edm::ESHandle<EcalPedestals> ecalPeds; 
  c.get<EcalPedestalsRcd>().get(ecalPeds);

  // ---------------------- 
  // loop over UncalibRecHits
  for (EcalUncalibratedRecHitCollection::const_iterator uncalibRecHit = EBUncalibRecHit->begin(); uncalibRecHit != EBUncalibRecHit->end() ; ++uncalibRecHit)
    {
      EBDetId EBid = EBDetId(uncalibRecHit->id());
      
      // general checks
      if (meEBUncalibRecHitsOccupancy_)  meEBUncalibRecHitsOccupancy_  -> Fill(EBid.ieta(), EBid.iphi());
      if (meEBUncalibRecHitsAmplitude_)  meEBUncalibRecHitsAmplitude_  -> Fill(uncalibRecHit->amplitude());
      if (meEBUncalibRecHitsPedestal_)   meEBUncalibRecHitsPedestal_   -> Fill(uncalibRecHit->pedestal());
      if (meEBUncalibRecHitsJitter_)     meEBUncalibRecHitsJitter_     -> Fill(uncalibRecHit->jitter());
      if (meEBUncalibRecHitsChi2_)       meEBUncalibRecHitsChi2_       -> Fill(uncalibRecHit->chi2());
      if (meEBUncalibRecHitsAmpFullMap_) meEBUncalibRecHitsAmpFullMap_ -> Fill(EBid.ieta(), EBid.iphi(), uncalibRecHit->amplitude()); 
      if (meEBUncalibRecHitsPedFullMap_) meEBUncalibRecHitsPedFullMap_ -> Fill(EBid.ieta(), EBid.iphi(), uncalibRecHit->pedestal()); 

      // general checks, with threshold at 3.5 GeV = 100 ADC counts
      if ( uncalibRecHit->amplitude() > 100 ) 
	{
	  if (meEBUncalibRecHitsOccupancyGt100adc_)  meEBUncalibRecHitsOccupancyGt100adc_  -> Fill(EBid.ieta(), EBid.iphi());
	  if (meEBUncalibRecHitsAmplitudeGt100adc_)  meEBUncalibRecHitsAmplitudeGt100adc_  -> Fill(uncalibRecHit->amplitude());
	  if (meEBUncalibRecHitsPedestalGt100adc_)   meEBUncalibRecHitsPedestalGt100adc_   -> Fill(uncalibRecHit->pedestal());
	  if (meEBUncalibRecHitsJitterGt100adc_)     meEBUncalibRecHitsJitterGt100adc_     -> Fill(uncalibRecHit->jitter());
	  if (meEBUncalibRecHitsChi2Gt100adc_)       meEBUncalibRecHitsChi2Gt100adc_       -> Fill(uncalibRecHit->chi2());
	}

      // supermodule maps
      int ic = EBid.ic();              
      int ie = (ic-1)/20 + 1;          
      int ip = (ic-1)%20 + 1;
      int ism = EBid.ism();
      float xie = ie - 0.5;
      float xip = ip - 0.5;      
      if( meEBUncalibRecHitPedMap_[ism-1] ) meEBUncalibRecHitPedMap_[ism-1]->Fill(xie, xip, uncalibRecHit->pedestal());
      if( meEBUncalibRecHitAmplMap_[ism-1] ) meEBUncalibRecHitAmplMap_[ism-1]->Fill(xie, xip, uncalibRecHit->amplitude());
      
      if ( ! skipDigis ) { 
        // find the rechit corresponding digi and the max sample
        EBDigiCollection::const_iterator myDigi = EBDigi->find(EBid);
        int sMax = -1;
        double eMax = 0.;
        if (myDigi != EBDigi->end()){
          for (unsigned int sample = 0 ; sample < myDigi->size(); ++sample){
            EcalMGPASample thisSample = (*myDigi)[sample];
            double analogSample = thisSample.adc();
            if ( eMax < analogSample ){
              eMax = analogSample;
              sMax = sample;
            }
          }
        }
        else
          continue;

        // ratio uncalibratedRecHit amplitude + ped / max energy digi  
        const EcalPedestals* myped = ecalPeds.product();
        EcalPedestalsMap::const_iterator it=myped->getMap().find( EBid );
        if( it != myped->getMap().end() ){
          
          if (eMax > (*it).mean_x1 + 5 * (*it).rms_x1 && eMax != 0 ) {//only real signal RecHit
            
            if ( meEBUncalibRecHitMaxSampleRatio_ ) meEBUncalibRecHitMaxSampleRatio_->Fill( (uncalibRecHit->amplitude()+uncalibRecHit->pedestal())/eMax);
            if ( meEBUncalibRecHitMaxSampleRatioGt100adc_ && (uncalibRecHit->amplitude()>100) ) meEBUncalibRecHitMaxSampleRatioGt100adc_->Fill( (uncalibRecHit->amplitude()+uncalibRecHit->pedestal())/eMax);
            LogDebug("EcalRecHitsTaskInfo") << "barrel, eMax = " << eMax << " Amplitude = " << uncalibRecHit->amplitude()+uncalibRecHit->pedestal();  
          }
          else
            continue;
        }
        else
          continue;
      }
      
    }  // loop over the UncalibratedRecHitCollection
}
