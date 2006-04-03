/*
 * \file EcalDigisValidation.cc
 *
 * $Date: 2006/03/23 10:57:04 $
 * $Revision: 1.5 $
 * \author F. Cossutti
 *
*/

#include <Validation/EcalDigis/interface/EcalDigisValidation.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include <DataFormats/EcalDetId/interface/ESDetId.h>

EcalDigisValidation::EcalDigisValidation(const ParameterSet& ps):
  HepMCLabel(ps.getUntrackedParameter("moduleLabelMC",string("PythiaSource"))),
  SimTkLabel(ps.getUntrackedParameter("moduleLabelTk",string("EmbdSimTrack"))),
  SimVtxLabel(ps.getUntrackedParameter("moduleLabelVtx",string("EmbdSimVertex"))){

 
  // DQM ROOT output
  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
 
  if ( outputFile_.size() != 0 ) {
    LogInfo("OutputInfo") << " Ecal Digi Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    LogInfo("OutputInfo") << " Ecal Digi Task histograms will NOT be saved";
  }
 
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
 
  meGunEnergy_ = 0;
  meGunEta_ = 0;   
  meGunPhi_ = 0;   

  meEBDigiOccupancy_ = 0;
  meEEDigiOccupancy_ = 0;

  meEBDigiADCGlobal_ = 0;
  meEEDigiADCGlobal_ = 0;

  for (int i = 0; i < 10 ; i++ ) {
    meEBDigiADCAnalog_[i] = 0;
    meEBDigiADCg1_[i] = 0;
    meEBDigiADCg6_[i] = 0;
    meEBDigiADCg12_[i] = 0;
    meEEDigiADCAnalog_[i] = 0;
    meEEDigiADCg1_[i] = 0;
    meEEDigiADCg6_[i] = 0;
    meEEDigiADCg12_[i] = 0;
    meEBDigiGain_[i] = 0;
    meEEDigiGain_[i] = 0;
  }

  for (int i = 0; i < 3 ; i++ ) {
    meESDigiADC_[i] = 0;
  }

  meEBPedestal_ = 0;
  meEEPedestal_ = 0;
                                 
  meEBMaximumgt100ADC_ = 0; 
  meEEMaximumgt100ADC_ = 0; 
                                 
  meEBMaximumgt10ADC_ = 0; 
  meEEMaximumgt10ADC_ = 0; 

  meEBDigiSimRatio_ = 0;
  meEEDigiSimRatio_ = 0;

  meEBDigiSimRatiogt10ADC_ = 0;
  meEEDigiSimRatiogt10ADC_ = 0;

  meEBDigiSimRatiogt100ADC_ = 0;
  meEEDigiSimRatiogt100ADC_ = 0;

  meEBnADCafterSwitch_ = 0;
  meEEnADCafterSwitch_ = 0;
 
  Char_t histo[20];
 
  
  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalDigiTask");
  
    sprintf (histo, "EcalDigiTask Gun Momentum" ) ;
    meGunEnergy_ = dbe_->book1D(histo, histo, 100, 0., 1000.);
  
    sprintf (histo, "EcalDigiTask Gun Eta" ) ;
    meGunEta_ = dbe_->book1D(histo, histo, 500, -2.5, 2.5);
  
    sprintf (histo, "EcalDigiTask Gun Phi" ) ;
    meGunPhi_ = dbe_->book1D(histo, histo, 360, 0., 360.);

    sprintf (histo, "EcalDigiTask Barrel occupancy" ) ;
    meEBDigiOccupancy_ = dbe_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);
  
    sprintf (histo, "EcalDigiTask Endcap occupancy" ) ;
    meEEDigiOccupancy_ = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    
    sprintf (histo, "EcalDigiTask Barrel global pulse shape" ) ;
    meEBDigiADCGlobal_ = dbe_->bookProfile(histo, histo, 10, 0, 10, 10000, 0., 1000.) ;
    
    sprintf (histo, "EcalDigiTask Endcap global pulse shape" ) ;
    meEEDigiADCGlobal_ = dbe_->bookProfile(histo, histo, 10, 0, 10, 10000, 0., 1000.) ;

    for (int i = 0; i < 10 ; i++ ) {

      sprintf (histo, "EcalDigiTask Barrel analog pulse %02d", i+1) ;
      meEBDigiADCAnalog_[i] = dbe_->book1D(histo, histo, 512, 0., 4096.);

      sprintf (histo, "EcalDigiTask Barrel ADC pulse %02d Gain 1", i+1) ;
      meEBDigiADCg1_[i] = dbe_->book1D(histo, histo, 512, 0., 4096);

      sprintf (histo, "EcalDigiTask Barrel ADC pulse %02d Gain 6", i+1) ;
      meEBDigiADCg6_[i] = dbe_->book1D(histo, histo, 512, 0., 4096);

      sprintf (histo, "EcalDigiTask Barrel ADC pulse %02d Gain 12", i+1) ;
      meEBDigiADCg12_[i] = dbe_->book1D(histo, histo, 512, 0., 4096);

      sprintf (histo, "EcalDigiTask Endcap analog pulse %02d", i+1) ;
      meEEDigiADCAnalog_[i] = dbe_->book1D(histo, histo, 512, 0., 4096.);

      sprintf (histo, "EcalDigiTask Endcap ADC pulse %02d Gain 1", i+1) ;
      meEEDigiADCg1_[i] = dbe_->book1D(histo, histo, 512, 0., 4096);

      sprintf (histo, "EcalDigiTask Endcap ADC pulse %02d Gain 6", i+1) ;
      meEEDigiADCg6_[i] = dbe_->book1D(histo, histo, 512, 0., 4096);

      sprintf (histo, "EcalDigiTask Endcap ADC pulse %02d Gain 12", i+1) ;
      meEEDigiADCg12_[i] = dbe_->book1D(histo, histo, 512, 0., 4096);

      sprintf (histo, "EcalDigiTask Barrel gain pulse %02d", i+1) ;
      meEBDigiGain_[i] = dbe_->book1D(histo, histo, 3, 0, 3);

      sprintf (histo, "EcalDigiTask Endcap gain pulse %02d", i+1) ;
      meEEDigiGain_[i] = dbe_->book1D(histo, histo, 3, 0, 3);
    }
    
    for ( int i = 0; i < 3 ; i++ ) {
      
      sprintf (histo, "EcalDigiTask Preshower ADC pulse %02d", i+1) ;
      meESDigiADC_[i] = dbe_->book1D(histo, histo, 512, 0., 4096.) ;
    }

    sprintf (histo, "EcalDigiTask Barrel pedestal for pre-sample" ) ;
    meEBPedestal_ = dbe_->book1D(histo, histo, 512, 0., 4096.) ;

    sprintf (histo, "EcalDigiTask Endcap pedestal for pre-sample" ) ;
    meEEPedestal_ = dbe_->book1D(histo, histo, 512, 0., 4096.) ;

    sprintf (histo, "EcalDigiTask Barrel maximum position gt 100 ADC" ) ;
    meEBMaximumgt100ADC_ = dbe_->book1D(histo, histo, 10, 0., 10.) ;

    sprintf (histo, "EcalDigiTask Endcap maximum position gt 100 ADC" ) ;
    meEEMaximumgt100ADC_ = dbe_->book1D(histo, histo, 10, 0., 10.) ;

    sprintf (histo, "EcalDigiTask Barrel maximum position gt 10 ADC" ) ;
    meEBMaximumgt10ADC_ = dbe_->book1D(histo, histo, 10, 0., 10.) ;

    sprintf (histo, "EcalDigiTask Endcap maximum position gt 10 ADC" ) ;
    meEEMaximumgt10ADC_ = dbe_->book1D(histo, histo, 10, 0., 10.) ;

    sprintf (histo, "EcalDigiTask Barrel maximum Digi over Sim ratio" ) ;
    meEBDigiSimRatio_ = dbe_->book1D(histo, histo, 100, 0., 2.) ;

    sprintf (histo, "EcalDigiTask Endcap maximum Digi over Sim ratio" ) ;
    meEEDigiSimRatio_ = dbe_->book1D(histo, histo, 100, 0., 2.) ;

    sprintf (histo, "EcalDigiTask Barrel maximum Digi over Sim ratio gt 10 ADC" ) ;
    meEBDigiSimRatiogt10ADC_ = dbe_->book1D(histo, histo, 100, 0., 2.) ;

    sprintf (histo, "EcalDigiTask Endcap maximum Digi over Sim ratio gt 10 ADC" ) ;
    meEEDigiSimRatiogt10ADC_ = dbe_->book1D(histo, histo, 100, 0., 2.) ;

    sprintf (histo, "EcalDigiTask Barrel maximum Digi over Sim ratio gt 100 ADC" ) ;
    meEBDigiSimRatiogt100ADC_ = dbe_->book1D(histo, histo, 100, 0., 2.) ;

    sprintf (histo, "EcalDigiTask Endcap maximum Digi over Sim ratio gt 100 ADC" ) ;
    meEEDigiSimRatiogt100ADC_ = dbe_->book1D(histo, histo, 100, 0., 2.) ;

    sprintf (histo, "EcalDigiTask Barrel ADC counts after gain switch" ) ;
    meEBnADCafterSwitch_ = dbe_->book1D(histo, histo, 10, 0., 10.) ;

    sprintf (histo, "EcalDigiTask Endcap ADC counts after gain switch" ) ;
    meEEnADCafterSwitch_ = dbe_->book1D(histo, histo, 10, 0., 10.) ;

  }
 
}

EcalDigisValidation::~EcalDigisValidation(){
 
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);

}

void EcalDigisValidation::beginJob(const EventSetup& c){

}

void EcalDigisValidation::endJob(){

}

void EcalDigisValidation::analyze(const Event& e, const EventSetup& c){

  LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();

  vector<EmbdSimTrack> theSimTracks;
  vector<EmbdSimVertex> theSimVertexes;

  Handle<HepMCProduct> MCEvt;
  Handle<EmbdSimTrackContainer> SimTk;
  Handle<EmbdSimVertexContainer> SimVtx;
  Handle<CrossingFrame> crossingFrame;
  Handle<EBDigiCollection> EcalDigiEB;
  Handle<EEDigiCollection> EcalDigiEE;
  Handle<ESDigiCollection> EcalDigiES;

  e.getByLabel(HepMCLabel, MCEvt);
  e.getByLabel(SimTkLabel,SimTk);
  e.getByLabel(SimVtxLabel,SimVtx);
  e.getByType(crossingFrame);
  e.getByType(EcalDigiEB);
  e.getByType(EcalDigiEE);
  e.getByType(EcalDigiES);

  theSimTracks.insert(theSimTracks.end(),SimTk->begin(),SimTk->end());
  theSimVertexes.insert(theSimVertexes.end(),SimVtx->begin(),SimVtx->end());

  for ( HepMC::GenEvent::particle_const_iterator p = MCEvt->GetEvent()->particles_begin();
        p != MCEvt->GetEvent()->particles_end(); ++p ) {

    Hep3Vector hmom = Hep3Vector((*p)->momentum().vect());
    double htheta = hmom.theta();
    double heta = -log(tan(htheta * 0.5));
    double hphi = hmom.phi();
    hphi = (hphi>=0) ? hphi : hphi+2*M_PI;
    hphi = hphi / M_PI * 180.;
    LogDebug("EventInfo") << "Particle gun type form MC = " << abs((*p)->pdg_id()) << "\n" << "Energy = "<< (*p)->momentum().e() << " Eta = " << heta << " Phi = " << hphi;

    if (meGunEnergy_) meGunEnergy_->Fill((*p)->momentum().e());
    if (meGunEta_) meGunEta_->Fill(heta);
    if (meGunPhi_) meGunPhi_->Fill(hphi);

  }

  for (vector<EmbdSimVertex>::iterator isimvtx = theSimVertexes.begin();
       isimvtx != theSimVertexes.end(); ++isimvtx){
    if (verbose_ ) {
      LogDebug("EventInfo") <<" Vertex position  x = "<<isimvtx->position().x() <<" y = "<<isimvtx->position().y() <<" z = "<< isimvtx->position().z();
    }
  }

  // BARREL

  // loop over simHits

  const std::string barrelHitsName ("EcalHitsEB") ;
  std::auto_ptr<MixCollection<PCaloHit> > 
    barrelHits (new MixCollection<PCaloHit>(crossingFrame.product (), barrelHitsName)) ;

  MapType ebSimMap;

  for (MixCollection<PCaloHit>::MixItr hitItr = barrelHits->begin () ;
       hitItr != barrelHits->end () ;
       ++hitItr) {
   
    EBDetId ebid = EBDetId(hitItr->id()) ;

    LogDebug("HitInfo") 
      <<" CaloHit " << hitItr->getName() << " DetID = "<<hitItr->id()<< "\n"	
      << "Energy = " << hitItr->energy() << " Time = " << hitItr->time() << "\n"
      << "EBDetId = " << ebid.ieta() << " " << ebid.iphi();

    uint32_t crystid = ebid.rawId();
    ebSimMap[crystid] += hitItr->energy();

  }

  // loop over Digis

  const EBDigiCollection * barrelDigi = EcalDigiEB.product () ;

  for (std::vector<EBDataFrame>::const_iterator digis = barrelDigi->begin () ;
       digis != barrelDigi->end () ;
       ++digis)
    {
    
      EBDetId ebid = digis->id () ;

      if (meEBDigiOccupancy_) meEBDigiOccupancy_->Fill( ebid.iphi(), ebid.ieta() );

      double Emax = 0. ;
      int Pmax = 0 ;
      std::vector<double> ebAnalogSignal ;
      std::vector<double> ebADCCounts ;
      std::vector<double> ebADCGains ;
      double pedestalPreSample = 0.;
      double pedestalPreSampleAnalog = 0.;
      int countsAfterGainSwitch = -1;
      double higherGain = 2.;
      int higherGainSample = 0;
      for (int sample = 0 ; sample < digis->size () ; ++sample)
        {
          ebADCCounts.push_back (digis->sample (sample).adc ()) ;
          ebADCGains.push_back (digis->sample (sample).gainId ()) ;
          ebAnalogSignal.push_back (ebADCCounts[sample]*gainConv_[(int)ebADCGains[sample]]*barrelADCtoGeV_);
          if (Emax < ebAnalogSignal[sample] ) {
            Emax = ebAnalogSignal[sample] ;
            Pmax = sample ;
          }
          if ( sample < 3 ) {
            pedestalPreSample += ebADCCounts[sample] ;
            pedestalPreSampleAnalog += ebADCCounts[sample]*gainConv_[(int)ebADCGains[sample]]*barrelADCtoGeV_ ;
          }
          if ( sample > 0 && ebADCGains[sample] < ebADCGains[sample-1] ) {
            higherGain = ebADCGains[sample];
            higherGainSample = sample;
            countsAfterGainSwitch = 1;
          }
          if ( higherGain < 2 && higherGainSample != sample && ebADCGains[sample] == higherGain) countsAfterGainSwitch++ ;
        }
      pedestalPreSample /= 3. ; 
      pedestalPreSampleAnalog /= 3. ; 

      LogDebug("DigiInfo") << "Barrel Digi for EBDetId = " << ebid.rawId() << " eta,phi " << ebid.ieta() << " " << ebid.iphi() ;
      for ( int i = 0; i < 10 ; i++ ) {
        LogDebug("DigiInfo") << "sample " << i << " ADC = " << ebADCCounts[i] << " gain = " << ebADCGains[i] << " Analog = " << ebAnalogSignal[i];
      }
      LogDebug("DigiInfo") << "Maximum energy = " << Emax << " in sample " << Pmax ;
      if ( countsAfterGainSwitch > 0 ) LogDebug("DigiInfo") << "Counts after switch " << countsAfterGainSwitch;
        
      for ( int i = 0 ; i < 10 ; i++ ) {
        if (meEBDigiADCGlobal_) meEBDigiADCGlobal_->Fill( i , ebAnalogSignal[i] ) ;
        if (meEBDigiADCAnalog_[i]) meEBDigiADCAnalog_[i]->Fill( ebAnalogSignal[i]*100. ) ;
        if ( ebADCGains[i] == 0 ) {
          if (meEBDigiADCg1_[i]) meEBDigiADCg1_[i]->Fill( ebADCCounts[i] ) ;
        }
        else if ( ebADCGains[i] == 1 ) {
          if (meEBDigiADCg6_[i]) meEBDigiADCg6_[i]->Fill( ebADCCounts[i] ) ;
        }
        else if ( ebADCGains[i] == 2 ) {
          if (meEBDigiADCg12_[i]) meEBDigiADCg12_[i]->Fill( ebADCCounts[i] ) ;
        }
        if (meEBDigiGain_[i]) meEBDigiGain_[i]->Fill( ebADCGains[i] ) ;
      }

      if (meEBPedestal_) meEBPedestal_->Fill ( pedestalPreSample ) ;
      if (meEBMaximumgt10ADC_ && ebSimMap[ebid.rawId()] > 10.*barrelADCtoGeV_) meEBMaximumgt10ADC_->Fill( Pmax ) ;
      if (meEBMaximumgt100ADC_ && ebSimMap[ebid.rawId()] > 100.*barrelADCtoGeV_) meEBMaximumgt100ADC_->Fill( Pmax ) ;
      if (meEBnADCafterSwitch_) meEBnADCafterSwitch_->Fill( countsAfterGainSwitch ) ;
        
      if ( ebSimMap[ebid.rawId()] != 0. ) {
        LogDebug("DigiInfo") << " Digi / Hit " << Emax << " " << ebSimMap[ebid.rawId()] << " gainConv " << gainConv_[(int)ebADCGains[Pmax]];
        double Erec = Emax - pedestalPreSampleAnalog;
        if ( meEBDigiSimRatio_ ) meEBDigiSimRatio_->Fill( Erec/ebSimMap[ebid.rawId()] ) ; 
        if ( ebSimMap[ebid.rawId()] > 10.*barrelADCtoGeV_  && meEBDigiSimRatiogt10ADC_  ) meEBDigiSimRatiogt10ADC_->Fill( Erec/ebSimMap[ebid.rawId()] );
        if ( ebSimMap[ebid.rawId()] > 100.*barrelADCtoGeV_  && meEBDigiSimRatiogt100ADC_  ) meEBDigiSimRatiogt100ADC_->Fill( Erec/ebSimMap[ebid.rawId()] );
      }
        
    } 

  // ENDCAP

  // loop over simHits

  const std::string endcapHitsName ("EcalHitsEE") ;
  std::auto_ptr<MixCollection<PCaloHit> > 
    endcapHits (new MixCollection<PCaloHit>(crossingFrame.product (), endcapHitsName)) ;

  MapType eeSimMap;

  for (MixCollection<PCaloHit>::MixItr hitItr = endcapHits->begin () ;
       hitItr != endcapHits->end () ;
       ++hitItr) {
   
    EEDetId eeid = EEDetId(hitItr->id()) ;

    LogDebug("HitInfo")
      <<" CaloHit " << hitItr->getName() << " DetID = "<<hitItr->id()<< "\n"
      << "Energy = " << hitItr->energy() << " Time = " << hitItr->time() << "\n"
      << "EEDetId side " << eeid.zside() << " = " << eeid.ix() << " " << eeid.iy() ;

    uint32_t crystid = eeid.rawId();
    eeSimMap[crystid] += hitItr->energy();

  }

  // loop over Digis

  const EEDigiCollection * endcapDigi = EcalDigiEE.product () ;

  for (std::vector<EEDataFrame>::const_iterator digis = endcapDigi->begin () ;
       digis != endcapDigi->end () ;
       ++digis)
    {
    
      EEDetId eeid = digis->id () ;

      if (meEEDigiOccupancy_) meEEDigiOccupancy_->Fill( eeid.ix(), eeid.iy() );

      double Emax = 0. ;
      int Pmax = 0 ;
      std::vector<double> eeAnalogSignal ;
      std::vector<double> eeADCCounts ;
      std::vector<double> eeADCGains ;
      double pedestalPreSample = 0.;
      double pedestalPreSampleAnalog = 0.;
      int countsAfterGainSwitch = -1;
      double higherGain = 2.;
      int higherGainSample = 0;
      for (int sample = 0 ; sample < digis->size () ; ++sample)
        {
          eeADCCounts.push_back (digis->sample (sample).adc ()) ;
          eeADCGains.push_back (digis->sample (sample).gainId ()) ;
          eeAnalogSignal.push_back (eeADCCounts[sample]*gainConv_[(int)eeADCGains[sample]]*endcapADCtoGeV_);
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
      LogDebug("DigiInfo") << "Maximum energy = " << Emax << " in sample " << Pmax ;
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
      if (meEEMaximumgt10ADC_ && eeSimMap[eeid.rawId()] > 10.*endcapADCtoGeV_) meEEMaximumgt10ADC_->Fill( Pmax ) ;
      if (meEEMaximumgt100ADC_ && eeSimMap[eeid.rawId()] > 100.*endcapADCtoGeV_) meEEMaximumgt100ADC_->Fill( Pmax ) ;
      if (meEEnADCafterSwitch_) meEEnADCafterSwitch_->Fill(countsAfterGainSwitch);

      if (eeSimMap[eeid.rawId()] != 0. ) {
        LogDebug("DigiInfo") << " Digi / Hit " << Emax << " " << eeSimMap[eeid.rawId()] << " gainConv " << gainConv_[(int)eeADCGains[Pmax]];
        double Erec = Emax - pedestalPreSampleAnalog;
        if ( meEEDigiSimRatio_) meEEDigiSimRatio_->Fill( Erec/eeSimMap[eeid.rawId()] ) ; 
        if ( eeSimMap[eeid.rawId()] > 10.*barrelADCtoGeV_  && meEEDigiSimRatiogt10ADC_  ) meEEDigiSimRatiogt10ADC_->Fill( Erec/eeSimMap[eeid.rawId()] );
        if ( eeSimMap[eeid.rawId()] > 100.*barrelADCtoGeV_  && meEEDigiSimRatiogt100ADC_  ) meEEDigiSimRatiogt100ADC_->Fill( Erec/eeSimMap[eeid.rawId()] );
      }

    } 

  // PRESHOWER

  // loop over Digis

  const ESDigiCollection * preshowerDigi = EcalDigiES.product () ;

  for (std::vector<ESDataFrame>::const_iterator digis = preshowerDigi->begin () ;
       digis != preshowerDigi->end () ;
       ++digis)
    {
       
      ESDetId esid = digis->id () ;
       
      std::vector<double> esADCCounts ;
      for (int sample = 0 ; sample < digis->size () ; ++sample)
        {
          esADCCounts.push_back (digis->sample (sample).adc ()) ;
        }
      if (verbose_) {
        LogDebug("DigiInfo") << "Preshower Digi for ESDetId: z side " << esid.zside() << "  plane " << esid.plane() << esid.six() << ',' << esid.siy() << ':' << esid.strip();
        for ( int i = 0; i < 3 ; i++ ) {
          LogDebug("DigiInfo") << "sample " << i << " ADC = " << esADCCounts[i];
        }
      }
       
      for ( int i = 0 ; i < 3 ; i++ ) {
        if (meESDigiADC_[i]) meESDigiADC_[i]->Fill( esADCCounts[i] ) ;
      }
       
    } 

}

                                                                                                                                                             
