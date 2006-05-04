/*
 * \file EcalEndcapDigisValidation.cc
 *
 * $Date: 2006/05/04 11:16:28 $
 * $Revision: 1.3 $
 * \author F. Cossutti
 *
*/

#include <Validation/EcalDigis/interface/EcalEndcapDigisValidation.h>

EcalEndcapDigisValidation::EcalEndcapDigisValidation(const ParameterSet& ps)
  {
 
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);
 
  if ( verbose_ ) {
    cout << " verbose switch is ON" << endl;
  } else {
    cout << " verbose switch is OFF" << endl;
  }
                                                                                                                                          
  dbe_ = 0;
                                                                                                                                          
  // get hold of back-end interface
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

  gainConv_[-1] = 0.;
  gainConv_[0] = 12.;
  gainConv_[1] = 6.;
  gainConv_[2] = 1.;
  barrelADCtoGeV_ = 0.035;
  endcapADCtoGeV_ = 0.06;
 
  meEEDigiOccupancyzp_ = 0;
  meEEDigiOccupancyzm_ = 0;

  meEEDigiADCGlobal_ = 0;

  for (int i = 0; i < 10 ; i++ ) {
    meEEDigiADCAnalog_[i] = 0;
    meEEDigiADCg1_[i] = 0;
    meEEDigiADCg6_[i] = 0;
    meEEDigiADCg12_[i] = 0;
    meEEDigiGain_[i] = 0;
  }

  meEEPedestal_ = 0;
                                 
  meEEMaximumgt100ADC_ = 0; 
                                 
  meEEMaximumgt10ADC_ = 0; 

  meEEnADCafterSwitch_ = 0;
 
  Char_t histo[20];
 
  
  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalDigiTask");
  
    sprintf (histo, "EcalDigiTask Endcap occupancy z+" ) ;
    meEEDigiOccupancyzp_ = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    
    sprintf (histo, "EcalDigiTask Endcap occupancy z-" ) ;
    meEEDigiOccupancyzm_ = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    
    sprintf (histo, "EcalDigiTask Endcap global pulse shape" ) ;
    meEEDigiADCGlobal_ = dbe_->bookProfile(histo, histo, 10, 0, 10, 10000, 0., 1000.) ;

    for (int i = 0; i < 10 ; i++ ) {

      sprintf (histo, "EcalDigiTask Endcap analog pulse %02d", i+1) ;
      meEEDigiADCAnalog_[i] = dbe_->book1D(histo, histo, 512, 0., 4096.);

      sprintf (histo, "EcalDigiTask Endcap ADC pulse %02d Gain 1", i+1) ;
      meEEDigiADCg1_[i] = dbe_->book1D(histo, histo, 512, 0., 4096);

      sprintf (histo, "EcalDigiTask Endcap ADC pulse %02d Gain 6", i+1) ;
      meEEDigiADCg6_[i] = dbe_->book1D(histo, histo, 512, 0., 4096);

      sprintf (histo, "EcalDigiTask Endcap ADC pulse %02d Gain 12", i+1) ;
      meEEDigiADCg12_[i] = dbe_->book1D(histo, histo, 512, 0., 4096);

      sprintf (histo, "EcalDigiTask Endcap gain pulse %02d", i+1) ;
      meEEDigiGain_[i] = dbe_->book1D(histo, histo, 3, 0, 3);
    }
    
    sprintf (histo, "EcalDigiTask Endcap pedestal for pre-sample" ) ;
    meEEPedestal_ = dbe_->book1D(histo, histo, 512, 0., 4096.) ;

    sprintf (histo, "EcalDigiTask Endcap maximum position gt 100 ADC" ) ;
    meEEMaximumgt100ADC_ = dbe_->book1D(histo, histo, 10, 0., 10.) ;

    sprintf (histo, "EcalDigiTask Endcap maximum position gt 10 ADC" ) ;
    meEEMaximumgt10ADC_ = dbe_->book1D(histo, histo, 10, 0., 10.) ;

    sprintf (histo, "EcalDigiTask Endcap ADC counts after gain switch" ) ;
    meEEnADCafterSwitch_ = dbe_->book1D(histo, histo, 10, 0., 10.) ;

  }
 
}

EcalEndcapDigisValidation::~EcalEndcapDigisValidation(){
 
}

void EcalEndcapDigisValidation::beginJob(const EventSetup& c){

}

void EcalEndcapDigisValidation::endJob(){

}

void EcalEndcapDigisValidation::analyze(const Event& e, const EventSetup& c){

  //LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();

  Handle<EEDigiCollection> EcalDigiEE;

  e.getByType(EcalDigiEE);

  // ENDCAP

  // loop over Digis

  const EEDigiCollection * endcapDigi = EcalDigiEE.product () ;

  std::vector<double> eeAnalogSignal ;
  std::vector<double> eeADCCounts ;
  std::vector<double> eeADCGains ;
  eeAnalogSignal.reserve(EEDataFrame::MAXSAMPLES);
  eeADCCounts.reserve(EEDataFrame::MAXSAMPLES);
  eeADCGains.reserve(EEDataFrame::MAXSAMPLES);

  for (std::vector<EEDataFrame>::const_iterator digis = endcapDigi->begin () ;
       digis != endcapDigi->end () ;
       ++digis)
    {
    
      EEDetId eeid = digis->id () ;

      if (eeid.zside() > 0 ) {
        if (meEEDigiOccupancyzp_) meEEDigiOccupancyzp_->Fill( eeid.ix(), eeid.iy() );
      }
      else if (eeid.zside() < 0 ) {
        if (meEEDigiOccupancyzm_) meEEDigiOccupancyzm_->Fill( eeid.ix(), eeid.iy() );
      }
      
      double Emax = 0. ;
      int Pmax = 0 ;
      double pedestalPreSample = 0.;
      double pedestalPreSampleAnalog = 0.;
      int countsAfterGainSwitch = -1;
      double higherGain = 2.;
      int higherGainSample = 0;

      for (int sample = 0 ; sample < digis->size () ; ++sample) {
        eeAnalogSignal[sample] = 0.;
        eeADCCounts[sample] = 0.;
        eeADCGains[sample] = -1.;
      }

      for (int sample = 0 ; sample < digis->size () ; ++sample)
        {
          eeADCCounts[sample] = (digis->sample (sample).adc ()) ;
          eeADCGains[sample] = (digis->sample (sample).gainId ()) ;
          eeAnalogSignal[sample] = (eeADCCounts[sample]*gainConv_[(int)eeADCGains[sample]]*endcapADCtoGeV_);
          if (Emax < eeAnalogSignal[sample] ) {
            Emax = eeAnalogSignal[sample] ;
            Pmax = sample ;
          }
          if ( sample < 3 ) {
            pedestalPreSample += eeADCCounts[sample] ;
            pedestalPreSampleAnalog += eeADCCounts[sample]*gainConv_[(int)eeADCGains[sample]]*endcapADCtoGeV_ ;
          }
          if ( sample > 0 && eeADCGains[sample] < eeADCGains[sample-1] ) {
            higherGain = eeADCGains[sample];
            higherGainSample = sample;
            countsAfterGainSwitch = 1;
          }
          if ( higherGain < 2 && higherGainSample != sample && eeADCGains[sample] == higherGain) countsAfterGainSwitch++ ;
        }
      pedestalPreSample /= 3. ; 
      pedestalPreSampleAnalog /= 3. ; 

      LogDebug("DigiInfo") << "Endcap Digi for EEDetId = " << eeid.rawId() << " x,y " << eeid.ix() << " " << eeid.iy() ;
      for ( int i = 0; i < 10 ; i++ ) {
        LogDebug("DigiInfo") << "sample " << i << " ADC = " << eeADCCounts[i] << " gain = " << eeADCGains[i] << " Analog = " << eeAnalogSignal[i] ;
      }
      LogDebug("DigiInfo") << "Maximum energy = " << Emax << " in sample " << Pmax << " Pedestal from pre-sample = " << pedestalPreSampleAnalog;
      if ( countsAfterGainSwitch > 0 ) LogDebug("DigiInfo") << "Counts after switch " << countsAfterGainSwitch;

      for ( int i = 0 ; i < 10 ; i++ ) {
        if (meEEDigiADCGlobal_) meEEDigiADCGlobal_->Fill( i , eeAnalogSignal[i] ) ;
        if (meEEDigiADCAnalog_[i]) meEEDigiADCAnalog_[i]->Fill( eeAnalogSignal[i]*100. ) ;
        if ( eeADCGains[i] == 0 ) {
          if (meEEDigiADCg1_[i]) meEEDigiADCg1_[i]->Fill( eeADCCounts[i] ) ;
        }
        else if ( eeADCGains[i] == 1 ) {
          if (meEEDigiADCg6_[i]) meEEDigiADCg6_[i]->Fill( eeADCCounts[i] ) ;
        }
        else if ( eeADCGains[i] == 2 ) {
          if (meEEDigiADCg12_[i]) meEEDigiADCg12_[i]->Fill( eeADCCounts[i] ) ;
        }
        if (meEEDigiGain_[i]) meEEDigiGain_[i]->Fill( eeADCGains[i] ) ;
      }

      if (meEEPedestal_) meEEPedestal_->Fill ( pedestalPreSample ) ;
      if (meEEMaximumgt10ADC_ && (Emax-pedestalPreSampleAnalog) > 10.*endcapADCtoGeV_) meEEMaximumgt10ADC_->Fill( Pmax ) ;
      if (meEEMaximumgt100ADC_ && (Emax-pedestalPreSampleAnalog) > 100.*endcapADCtoGeV_) meEEMaximumgt100ADC_->Fill( Pmax ) ;
      if (meEEnADCafterSwitch_) meEEnADCafterSwitch_->Fill(countsAfterGainSwitch);
      
    } 

}

                                                                                                                                                             
