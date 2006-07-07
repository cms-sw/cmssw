/*
 * \file EcalEndcapRecHitsValidation.cc
 *
 * $Date: 2006/06/29 11:07:41 $
 * \author C. Rovelli
 *
 */

#include <Validation/EcalRecHits/interface/EcalEndcapRecHitsValidation.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>

EcalEndcapRecHitsValidation::EcalEndcapRecHitsValidation(const ParameterSet& ps){

  // ---------------------- 
  digiProducer_              = ps.getParameter<std::string>("digiProducer");
  uncalibrecHitProducer_     = ps.getParameter<std::string>("uncalibrecHitProducer");
  EEdigiCollection_          = ps.getParameter<std::string>("EEdigiCollection");
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
  meEEUncalibRecHitsOccupancyPlus_         = 0;
  meEEUncalibRecHitsOccupancyMinus_        = 0;
  meEEUncalibRecHitsAmplitude_             = 0;
  meEEUncalibRecHitsPedestal_              = 0;
  meEEUncalibRecHitsJitter_                = 0;
  meEEUncalibRecHitsChi2_                  = 0;
  meEEUncalibRecHitMaxSampleRatio_         = 0;
  meEEUncalibRecHitsOccupancyPlusGt60adc_  = 0;
  meEEUncalibRecHitsOccupancyMinusGt60adc_ = 0;
  meEEUncalibRecHitsAmplitudeGt60adc_      = 0;
  meEEUncalibRecHitsPedestalGt60adc_       = 0;
  meEEUncalibRecHitsJitterGt60adc_         = 0;
  meEEUncalibRecHitsChi2Gt60adc_           = 0;
  meEEUncalibRecHitMaxSampleRatioGt60adc_  = 0;
  meEEUncalibRecHitsAmpFullMap_            = 0;
  meEEUncalibRecHitsPedFullMap_            = 0;

  // ---------------------- 
  Char_t histo[200];
   
  if ( dbe_ ) 
    {
      dbe_->setCurrentFolder("EcalEndcapRecHitsTask");
      
      sprintf (histo, "EE+ Occupancy" );  
      meEEUncalibRecHitsOccupancyPlus_ = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);

      sprintf (histo, "EE- Occupancy" );  
      meEEUncalibRecHitsOccupancyMinus_ = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
      
      sprintf (histo, "EE Amplitude" );
      meEEUncalibRecHitsAmplitude_ = dbe_->book1D(histo, histo, 200, 0., 4000.);
      
      sprintf (histo, "EE Pedestal" );
      meEEUncalibRecHitsPedestal_ = dbe_->book1D(histo, histo, 50, 190., 210.);
      
      sprintf (histo, "EE Jitter" );
      meEEUncalibRecHitsJitter_ = dbe_->book1D(histo, histo, 100, 0., 100.);
      
      sprintf (histo, "EE Chi2" );
      meEEUncalibRecHitsChi2_ = dbe_->book1D(histo, histo, 100, 0., 100.);

      sprintf (histo, "EE RecHit Max Sample Ratio"); 
      meEEUncalibRecHitMaxSampleRatio_ = dbe_->book1D(histo, histo, 80, 0.95, 1.05);
      
      sprintf (histo, "EE+ Occupancy gt 60 adc counts" );  
      meEEUncalibRecHitsOccupancyPlusGt60adc_ = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);

      sprintf (histo, "EE- Occupancy gt 60 adc counts" );  
      meEEUncalibRecHitsOccupancyMinusGt60adc_ = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);

      sprintf (histo, "EE Amplitude gt 60 adc counts" );
      meEEUncalibRecHitsAmplitudeGt60adc_ = dbe_->book1D(histo, histo, 200, 0., 4000.);
      
      sprintf (histo, "EE Pedestal gt 60 adc counts" );
      meEEUncalibRecHitsPedestalGt60adc_ = dbe_->book1D(histo, histo, 50, 190., 210.);
      
      sprintf (histo, "EE Jitter gt 60 adc counts" );
      meEEUncalibRecHitsJitterGt60adc_ = dbe_->book1D(histo, histo, 100, 0., 100.);
      
      sprintf (histo, "EE Chi2 gt 60 adc counts" );
      meEEUncalibRecHitsChi2Gt60adc_ = dbe_->book1D(histo, histo, 100, 0., 100.);

      sprintf (histo, "EE RecHit Max Sample Ratio gt 60 adc counts"); 
      meEEUncalibRecHitMaxSampleRatioGt60adc_ = dbe_->book1D(histo, histo, 80, 0.95, 1.05);

      sprintf (histo, "EE Amplitude Full Map");
      meEEUncalibRecHitsAmpFullMap_ = dbe_->bookProfile2D(histo, histo, 100, 0., 100., 100, 0., 100., 200, 0., 4000.);

      sprintf (histo, "EE Pedestal Full Map");
      meEEUncalibRecHitsPedFullMap_ = dbe_->bookProfile2D(histo, histo, 100, 0., 100., 100, 0., 100., 50, 194., 201.);
    }
}

EcalEndcapRecHitsValidation::~EcalEndcapRecHitsValidation(){   

}

void EcalEndcapRecHitsValidation::beginJob(const EventSetup& c){  

}

void EcalEndcapRecHitsValidation::endJob(){

}

void EcalEndcapRecHitsValidation::analyze(const Event& e, const EventSetup& c){

  Handle< EEDigiCollection > EcalDigiEE;
  try {
    e.getByLabel( digiProducer_, EcalDigiEE);
  } catch ( std::exception& ex ) {
    edm::LogError("EcalRecHitsTaskError") << "Error! can't get the Digis " << std::endl;
  }

  Handle< EEUncalibratedRecHitCollection > EcalUncalibRecHitEE;
  try {
    e.getByLabel( uncalibrecHitProducer_, EEuncalibrechitCollection_, EcalUncalibRecHitEE);
  } catch ( std::exception& ex ) {
    edm::LogError("EcalRecHitsTaskError") << "Error! can't get the product " << EEuncalibrechitCollection_.c_str() << std::endl;
  }

  edm::ESHandle<EcalPedestals> ecalPeds; 
  try {
    c.get<EcalPedestalsRcd>().get(ecalPeds);
  } catch ( std::exception& ex ) {
    edm::LogError("EcalRecHitsTaskError") << "Error! can't get the Ecal pedestals" << std::endl;
  }

  // ---------------------- 
  // loop over UncalibRecHits
  const EEDigiCollection *EEDigi = EcalDigiEE.product();
  const EEUncalibratedRecHitCollection *EEUncalibRecHit = EcalUncalibRecHitEE.product() ;
  
  for (EcalUncalibratedRecHitCollection::const_iterator uncalibRecHit = EEUncalibRecHit->begin(); uncalibRecHit != EEUncalibRecHit->end() ; ++uncalibRecHit)
    {
      EEDetId EEid = EEDetId(uncalibRecHit->id());

      int mySide = EEid.zside();      

      // general checks
      if (mySide > 0) { if (meEEUncalibRecHitsOccupancyPlus_)  meEEUncalibRecHitsOccupancyPlus_  ->Fill(EEid.ix(), EEid.iy()); }
      if (mySide < 0) { if (meEEUncalibRecHitsOccupancyMinus_) meEEUncalibRecHitsOccupancyMinus_ ->Fill(EEid.ix(), EEid.iy()); }
      if (meEEUncalibRecHitsAmplitude_)  meEEUncalibRecHitsAmplitude_  -> Fill(uncalibRecHit->amplitude());
      if (meEEUncalibRecHitsPedestal_)   meEEUncalibRecHitsPedestal_   -> Fill(uncalibRecHit->pedestal());
      if (meEEUncalibRecHitsJitter_)     meEEUncalibRecHitsJitter_     -> Fill(uncalibRecHit->jitter());
      if (meEEUncalibRecHitsChi2_)       meEEUncalibRecHitsChi2_       -> Fill(uncalibRecHit->chi2());
      if (meEEUncalibRecHitsAmpFullMap_) meEEUncalibRecHitsAmpFullMap_ -> Fill(EEid.ix(), EEid.iy(), uncalibRecHit->amplitude()); 
      if (meEEUncalibRecHitsPedFullMap_) meEEUncalibRecHitsPedFullMap_ -> Fill(EEid.ix(), EEid.iy(), uncalibRecHit->pedestal()); 


      // general checks, with threshold at 60 ADC counts
      if ( uncalibRecHit->amplitude() > 60 ) 
	{
	  if (mySide > 0) { if (meEEUncalibRecHitsOccupancyPlusGt60adc_)  meEEUncalibRecHitsOccupancyPlusGt60adc_ ->Fill(EEid.ix(), EEid.iy()); }
	  if (mySide < 0) { if (meEEUncalibRecHitsOccupancyMinusGt60adc_) meEEUncalibRecHitsOccupancyMinusGt60adc_->Fill(EEid.ix(), EEid.iy()); }
	  if (meEEUncalibRecHitsAmplitudeGt60adc_)  meEEUncalibRecHitsAmplitudeGt60adc_  -> Fill(uncalibRecHit->amplitude());
	  if (meEEUncalibRecHitsPedestalGt60adc_)   meEEUncalibRecHitsPedestalGt60adc_   -> Fill(uncalibRecHit->pedestal());
	  if (meEEUncalibRecHitsJitterGt60adc_)     meEEUncalibRecHitsJitterGt60adc_     -> Fill(uncalibRecHit->jitter());
	  if (meEEUncalibRecHitsChi2Gt60adc_)       meEEUncalibRecHitsChi2Gt60adc_       -> Fill(uncalibRecHit->chi2());
	}


      // Find the rechit corresponding digi
      EEDigiCollection::const_iterator myDigi = EEDigi->find(EEid);
      int sMax    = -1;
      double eMax = 0.;
      if (myDigi != EEDigi->end())
	{
	  for (int sample = 0; sample < myDigi->size(); ++sample)
	    {
	      double analogSample = myDigi->sample(sample).adc();
	      if ( eMax < analogSample )
		{
		  eMax = analogSample;
		  sMax = sample;
		}
	    }
	}
      else
	continue;
      
      // ratio uncalibratedRecHit amplitude + ped / max energy digi  
      const EcalPedestals* myped = ecalPeds.product();
      std::map<const unsigned int,EcalPedestals::Item>::const_iterator it=myped->m_pedestals.find(EEid.rawId());
      if( it != myped->m_pedestals.end() )
	{
	  if (eMax > it->second.mean_x1 + 5 * it->second.rms_x1 ) //only real signal RecHit
	    {
	      if ( meEEUncalibRecHitMaxSampleRatio_ )
		{ meEEUncalibRecHitMaxSampleRatio_->Fill( (uncalibRecHit->amplitude()+uncalibRecHit->pedestal())/eMax); }

	      if ( meEEUncalibRecHitMaxSampleRatioGt60adc_ && (uncalibRecHit->amplitude() > 60) ) 
		{ meEEUncalibRecHitMaxSampleRatioGt60adc_->Fill( (uncalibRecHit->amplitude()+uncalibRecHit->pedestal())/eMax); }

	      LogInfo("EcalRecHitsTaskInfo") << "endcap, eMax = " << eMax << " Amplitude = " << uncalibRecHit->amplitude()+uncalibRecHit->pedestal();  
	    }
	  else
	    continue;
	}
      else
	continue;


    }  // loop over the UncalibratedRecHitCollection

}
