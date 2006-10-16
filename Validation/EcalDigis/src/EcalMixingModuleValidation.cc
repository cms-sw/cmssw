/*
 * \file EcalMixingModuleValidation.cc
 *
 * $Date: 2006/10/05 13:21:05 $
 * $Revision: 1.15 $
 * \author F. Cossutti
 *
*/

#include <Validation/EcalDigis/interface/EcalMixingModuleValidation.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include <DataFormats/EcalDetId/interface/ESDetId.h>
#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"

EcalMixingModuleValidation::EcalMixingModuleValidation(const ParameterSet& ps):
  HepMCLabel(ps.getParameter<std::string>("moduleLabelMC")),
  EBdigiCollection_(ps.getParameter<edm::InputTag>("EBdigiCollection")),
  EEdigiCollection_(ps.getParameter<edm::InputTag>("EEdigiCollection")),
  ESdigiCollection_(ps.getParameter<edm::InputTag>("ESdigiCollection")){


  // needed for MixingModule checks

  double simHitToPhotoelectronsBarrel = ps.getParameter<double>("simHitToPhotoelectronsBarrel");
  double simHitToPhotoelectronsEndcap = ps.getParameter<double>("simHitToPhotoelectronsEndcap");
  double photoelectronsToAnalogBarrel = ps.getParameter<double>("photoelectronsToAnalogBarrel");
  double photoelectronsToAnalogEndcap = ps.getParameter<double>("photoelectronsToAnalogEndcap");
  double samplingFactor = ps.getParameter<double>("samplingFactor");
  double timePhase = ps.getParameter<double>("timePhase");
  int readoutFrameSize = ps.getParameter<int>("readoutFrameSize");
  int binOfMaximum = ps.getParameter<int>("binOfMaximum");
  bool doPhotostatistics = ps.getParameter<bool>("doPhotostatistics");
  bool syncPhase = ps.getParameter<bool>("syncPhase");

  doPhotostatistics = false;
    
  theParameterMap = new EcalSimParameterMap(simHitToPhotoelectronsBarrel, simHitToPhotoelectronsEndcap, 
                                            photoelectronsToAnalogBarrel, photoelectronsToAnalogEndcap, 
                                            samplingFactor, timePhase, readoutFrameSize, binOfMaximum,
                                            doPhotostatistics, syncPhase);
  theEcalShape = new EcalShape(timePhase);

  theEcalResponse = new CaloHitResponse(theParameterMap, theEcalShape);

  theMinBunch = -10;
  theMaxBunch = 10;
    
 
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

  gainConv_[0] = 0.;
  gainConv_[1] = 1.;
  gainConv_[2] = 2.;
  gainConv_[3] = 12.;
  barrelADCtoGeV_ = 0.035;
  endcapADCtoGeV_ = 0.06;
 
  meEBDigiMixRatiogt100ADC_ = 0;
  meEEDigiMixRatiogt100ADC_ = 0;
    
  meEBDigiMixRatioOriggt50pc_ = 0;
  meEEDigiMixRatioOriggt40pc_ = 0;
    
  meEBbunchCrossing_ = 0;
  meEEbunchCrossing_ = 0;
  meESbunchCrossing_ = 0;
    
  for ( int i = 0 ; i < nBunch ; i++ ) {
    meEBBunchShape_[i] = 0;
    meEEBunchShape_[i] = 0;
    meESBunchShape_[i] = 0;
  }

  meEBShape_ = 0;
  meEEShape_ = 0;
  meESShape_ = 0;

  meEBShapeRatio_ = 0;
  meEEShapeRatio_ = 0;
  meESShapeRatio_ = 0;
    

  Char_t histo[20];
 
  
  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalDigiTask");
  
    sprintf (histo, "EcalDigiTask Barrel maximum Digi over sim signal ratio gt 100 ADC" ) ;
    meEBDigiMixRatiogt100ADC_ = dbe_->book1D(histo, histo, 200, 0., 40.) ;
      
    sprintf (histo, "EcalDigiTask Endcap maximum Digi over sim signal ratio gt 100 ADC" ) ;
    meEEDigiMixRatiogt100ADC_ = dbe_->book1D(histo, histo, 200, 0., 40.) ;
      
    sprintf (histo, "EcalDigiTask Barrel maximum Digi over sim signal ratio signal gt 50pc gun" ) ;
    meEBDigiMixRatioOriggt50pc_ = dbe_->book1D(histo, histo, 100, 0., 20.) ;
      
    sprintf (histo, "EcalDigiTask Endcap maximum Digi over sim signal ratio signal gt 40pc gun" ) ;
    meEEDigiMixRatioOriggt40pc_ = dbe_->book1D(histo, histo, 100, 0., 20.) ;
      
    sprintf (histo, "EcalDigiTask Barrel bunch crossing" ) ;
    meEBbunchCrossing_ = dbe_->book1D(histo, histo, 20, -10., 10.) ;
      
    sprintf (histo, "EcalDigiTask Endcap bunch crossing" ) ;
    meEEbunchCrossing_ = dbe_->book1D(histo, histo, 20, -10., 10.) ;
      
    sprintf (histo, "EcalDigiTask Preshower bunch crossing" ) ;
    meESbunchCrossing_ = dbe_->book1D(histo, histo, 20, -10., 10.) ;

    for ( int i = 0 ; i < nBunch ; i++ ) {

      sprintf (histo, "EcalDigiTask Barrel shape bunch crossing %02d", i-10 );
      meEBBunchShape_[i] = dbe_->bookProfile(histo, histo, 10, 0, 10, 4000, 0., 400.);

      sprintf (histo, "EcalDigiTask Endcap shape bunch crossing %02d", i-10 );
      meEEBunchShape_[i] = dbe_->bookProfile(histo, histo, 10, 0, 10, 4000, 0., 400.);

      sprintf (histo, "EcalDigiTask Preshower shape bunch crossing %02d", i-10 );
      meESBunchShape_[i] = dbe_->bookProfile(histo, histo, 3, 0, 3, 4000, 0., 400.);
                        
    }

    sprintf (histo, "EcalDigiTask Barrel shape digi");
    meEBShape_ = dbe_->bookProfile(histo, histo, 10, 0, 10, 4000, 0., 400.);

    sprintf (histo, "EcalDigiTask Endcap shape digi");
    meEEShape_ = dbe_->bookProfile(histo, histo, 10, 0, 10, 4000, 0., 400.);

    sprintf (histo, "EcalDigiTask Preshower shape digi");
    meESShape_ = dbe_->bookProfile(histo, histo, 3, 0, 3, 4000, 0., 400.);

    sprintf (histo, "EcalDigiTask Barrel shape digi ratio");
    meEBShapeRatio_ = dbe_->book1D(histo, histo, 10, 0, 10.);

    sprintf (histo, "EcalDigiTask Endcap shape digi ratio");
    meEEShapeRatio_ = dbe_->book1D(histo, histo, 10, 0, 10.);

    sprintf (histo, "EcalDigiTask Preshower shape digi ratio");
    meESShapeRatio_ = dbe_->book1D(histo, histo, 3, 0, 3.);
     
  }
 
}

EcalMixingModuleValidation::~EcalMixingModuleValidation(){}

void EcalMixingModuleValidation::beginJob(const EventSetup& c){

  checkCalibrations(c);

}

void EcalMixingModuleValidation::endJob(){

  // add shapes for each bunch crossing and divide the digi by the result
  
  std::vector<MonitorElement *> theBunches;
  for ( int i = 0 ; i < nBunch ; i++ ) {
    theBunches.push_back(meEBBunchShape_[i]);
  }
  bunchSumTest(theBunches , meEBShape_ , meEBShapeRatio_ , EcalDataFrame::MAXSAMPLES);
  
  theBunches.clear();
  
  for ( int i = 0 ; i < nBunch ; i++ ) {
    theBunches.push_back(meEEBunchShape_[i]);
  }
  bunchSumTest(theBunches , meEEShape_ , meEEShapeRatio_ , EcalDataFrame::MAXSAMPLES);
  
}

void EcalMixingModuleValidation::bunchSumTest(std::vector<MonitorElement *> & theBunches, MonitorElement* & theTotal, MonitorElement* & theRatio, int nSample)
{

  std::vector<double> bunchSum;
  bunchSum.reserve(nSample);
  std::vector<double> bunchSumErro;
  bunchSumErro.reserve(nSample);
  std::vector<double> total;
  total.reserve(nSample);
  std::vector<double> totalErro;
  totalErro.reserve(nSample);
  std::vector<double> ratio;
  ratio.reserve(nSample);
  std::vector<double> ratioErro;
  ratioErro.reserve(nSample);


  for ( int iEl = 0 ; iEl < nSample ; iEl++ ) {
    bunchSum[iEl] = 0.;
    bunchSumErro[iEl] = 0.;
    total[iEl] = 0.;
    totalErro[iEl] = 0.;
    ratio[iEl] = 0.;
    ratioErro[iEl] = 0.;
  }

  for ( int iSample = 0 ; iSample < nSample ; iSample++ ) { 

    total[iSample] += theTotal->getBinContent(iSample+1);
    totalErro[iSample] += theTotal->getBinError(iSample+1);

    for ( int iBunch = theMinBunch; iBunch <= theMaxBunch; iBunch++ ) {

      int iHisto = iBunch - theMinBunch;

      bunchSum[iSample] += theBunches[iHisto]->getBinContent(iSample+1);
      bunchSumErro[iSample] += pow(theBunches[iHisto]->getBinError(iSample+1),2);

    }
    bunchSumErro[iSample] = sqrt(bunchSumErro[iSample]);

    if ( bunchSum[iSample] > 0. ) { 
      ratio[iSample] = total[iSample]/bunchSum[iSample];
      ratioErro[iSample] = sqrt(pow(totalErro[iSample]/bunchSum[iSample],2)+
                                pow((total[iSample]*bunchSumErro[iSample])/(bunchSum[iSample]*bunchSum[iSample]),2));
    }

    std::cout << " Sample = " << iSample << " Total = " << total[iSample] << " +- " << totalErro[iSample] << "\n" 
              << " Sum   = " << bunchSum[iSample] << " +- " << bunchSumErro[iSample] << "\n" 
              << " Ratio = " << ratio[iSample] << " +- " << ratioErro[iSample] << std::endl;
      
    theRatio->setBinContent(iSample+1, (float)ratio[iSample]);
    theRatio->setBinError(iSample+1, (float)ratioErro[iSample]);

  }

} 

void EcalMixingModuleValidation::analyze(const Event& e, const EventSetup& c){

  //LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();

  vector<SimTrack> theSimTracks;
  vector<SimVertex> theSimVertexes;

  Handle<HepMCProduct> MCEvt;
  Handle<CrossingFrame> crossingFrame;
  Handle<EBDigiCollection> EcalDigiEB;
  Handle<EEDigiCollection> EcalDigiEE;
  Handle<ESDigiCollection> EcalDigiES;

  e.getByLabel(HepMCLabel, MCEvt);
  e.getByType(crossingFrame);

  const EBDigiCollection* EBdigis =0;
  const EEDigiCollection* EEdigis =0;
  const ESDigiCollection* ESdigis =0;

  bool isBarrel = true;
  try {
    e.getByLabel( EBdigiCollection_, EcalDigiEB );
    EBdigis = EcalDigiEB.product();
    LogDebug("DigiInfo") << "total # EBdigis: " << EBdigis->size() ;
    if ( EBdigis->size() == 0 ) isBarrel = false;
  } catch ( cms::Exception &e ) { isBarrel = false; }
  bool isEndcap = true;
  try {
    e.getByLabel( EEdigiCollection_, EcalDigiEE );
    EEdigis = EcalDigiEE.product();
    LogDebug("DigiInfo") << "total # EEdigis: " << EEdigis->size() ;
    if ( EEdigis->size() == 0 ) isEndcap = false;
  } catch ( cms::Exception &e ) { isEndcap = false; }
  bool isPreshower = true;
  try {
    e.getByLabel( ESdigiCollection_, EcalDigiES );
    ESdigis = EcalDigiES.product();
    LogDebug("DigiInfo") << "total # ESdigis: " << ESdigis->size() ;
    if ( ESdigis->size() == 0 ) isPreshower = false;
  } catch ( cms::Exception &e ) { isPreshower = false; }

  double theGunEnergy = 0.;
  for ( HepMC::GenEvent::particle_const_iterator p = MCEvt->GetEvent()->particles_begin();
        p != MCEvt->GetEvent()->particles_end(); ++p ) {

    Hep3Vector hmom = Hep3Vector((*p)->momentum().vect());
    theGunEnergy = (*p)->momentum().e();

  }

  // BARREL

  // loop over simHits

  if ( isBarrel ) {

    const std::string barrelHitsName ("EcalHitsEB") ;
    std::auto_ptr<MixCollection<PCaloHit> > 
      barrelHits (new MixCollection<PCaloHit>(crossingFrame.product (), barrelHitsName)) ;
    
    MapType ebSignalSimMap;

    double ebSimThreshold = 0.5*theGunEnergy;
    
    for (MixCollection<PCaloHit>::MixItr hitItr = barrelHits->begin () ;
         hitItr != barrelHits->end () ;
         ++hitItr) {
      
      EBDetId ebid = EBDetId(hitItr->id()) ;
      
      LogDebug("HitInfo") 
        << " CaloHit " << hitItr->getName() << "\n" 
        << " DetID = "<<hitItr->id()<< " EBDetId = " << ebid.ieta() << " " << ebid.iphi() << "\n"	
        << " Time = " << hitItr->time() << " Event id. = " << hitItr->eventId().rawId() << "\n"
        << " Track Id = " << hitItr->geantTrackId() << "\n"
        << " Energy = " << hitItr->energy();

      uint32_t crystid = ebid.rawId();

      if ( hitItr->eventId().rawId() == 0 ) ebSignalSimMap[crystid] += hitItr->energy();
      
      if ( meEBbunchCrossing_ ) meEBbunchCrossing_->Fill(hitItr->eventId().bunchCrossing()); 
      
    }
    
    // loop over Digis
    
    const EBDigiCollection * barrelDigi = EcalDigiEB.product () ;
    
    std::vector<double> ebAnalogSignal ;
    std::vector<double> ebADCCounts ;
    std::vector<double> ebADCGains ;
    ebAnalogSignal.reserve(EBDataFrame::MAXSAMPLES);
    ebADCCounts.reserve(EBDataFrame::MAXSAMPLES);
    ebADCGains.reserve(EBDataFrame::MAXSAMPLES);
    
    for (std::vector<EBDataFrame>::const_iterator digis = barrelDigi->begin () ;
         digis != barrelDigi->end () ;
         ++digis)
      {
        
        EBDetId ebid = digis->id () ;
        
        double Emax = 0. ;
        int Pmax = 0 ;
        double pedestalPreSample = 0.;
        double pedestalPreSampleAnalog = 0.;
        
        for (int sample = 0 ; sample < digis->size () ; ++sample) {
          ebAnalogSignal[sample] = 0.;
          ebADCCounts[sample] = 0.;
          ebADCGains[sample] = -1.;
        }
        
        for (int sample = 0 ; sample < digis->size () ; ++sample)
          {
            ebADCCounts[sample] = (digis->sample (sample).adc ()) ;
            ebADCGains[sample] = (digis->sample (sample).gainId ()) ;
            ebAnalogSignal[sample] = (ebADCCounts[sample]*gainConv_[(int)ebADCGains[sample]]*barrelADCtoGeV_);
            if (Emax < ebAnalogSignal[sample] ) {
              Emax = ebAnalogSignal[sample] ;
              Pmax = sample ;
            }
            if ( sample < 3 ) {
              pedestalPreSample += ebADCCounts[sample] ;
              pedestalPreSampleAnalog += ebADCCounts[sample]*gainConv_[(int)ebADCGains[sample]]*barrelADCtoGeV_ ;
            }
            LogDebug("DigiInfo") << "EB sample " << sample << " ADC counts = " << ebADCCounts[sample] << " Gain Id = " << ebADCGains[sample] << " Analog eq = " << ebAnalogSignal[sample];
          }
        pedestalPreSample /= 3. ; 
        pedestalPreSampleAnalog /= 3. ; 
        double Erec = Emax - pedestalPreSampleAnalog*gainConv_[(int)ebADCGains[Pmax]];
        
        if ( ebSignalSimMap[ebid.rawId()] != 0. ) {
          LogDebug("DigiInfo") << " Digi / Signal Hit = " << Erec << " / " << ebSignalSimMap[ebid.rawId()] << " gainConv " << gainConv_[(int)ebADCGains[Pmax]];
          if ( Erec > 100.*barrelADCtoGeV_  && meEBDigiMixRatiogt100ADC_  ) meEBDigiMixRatiogt100ADC_->Fill( Erec/ebSignalSimMap[ebid.rawId()] );
          if ( ebSignalSimMap[ebid.rawId()] > ebSimThreshold  && meEBDigiMixRatioOriggt50pc_ ) meEBDigiMixRatioOriggt50pc_->Fill( Erec/ebSignalSimMap[ebid.rawId()] );
          if ( ebSignalSimMap[ebid.rawId()] > ebSimThreshold  && meEBShape_ ) {
            for ( int i = 0; i < 10 ; i++ ) {
              meEBShape_->Fill(i, ebAnalogSignal[i] );
            }
          }
        }
        
      } 
    
    computeCrystalBunchDigi(c, *barrelHits, ebSignalSimMap, isBarrel, ebSimThreshold);
  }
  
  
  // ENDCAP

  // loop over simHits

  if ( isEndcap ) {

    const std::string endcapHitsName ("EcalHitsEE") ;
    std::auto_ptr<MixCollection<PCaloHit> > 
      endcapHits (new MixCollection<PCaloHit>(crossingFrame.product (), endcapHitsName)) ;
    
    MapType eeSignalSimMap;

    double eeSimThreshold = 0.4*theGunEnergy;
    
    for (MixCollection<PCaloHit>::MixItr hitItr = endcapHits->begin () ;
         hitItr != endcapHits->end () ;
         ++hitItr) {
      
      EEDetId eeid = EEDetId(hitItr->id()) ;
      
      LogDebug("HitInfo") 
        << " CaloHit " << hitItr->getName() << "\n" 
        << " DetID = "<<hitItr->id()<< " EEDetId side = " << eeid.zside() << " = " << eeid.ix() << " " << eeid.iy() << "\n"
        << " Time = " << hitItr->time() << " Event id. = " << hitItr->eventId().rawId() << "\n"
        << " Track Id = " << hitItr->geantTrackId() << "\n"
        << " Energy = " << hitItr->energy();
      
      uint32_t crystid = eeid.rawId();

      if ( hitItr->eventId().rawId() == 0 ) eeSignalSimMap[crystid] += hitItr->energy();
      
      if ( meEEbunchCrossing_ ) meEEbunchCrossing_->Fill(hitItr->eventId().bunchCrossing()); 

    }
    
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
        
        double Emax = 0. ;
        int Pmax = 0 ;
        double pedestalPreSample = 0.;
        double pedestalPreSampleAnalog = 0.;
        
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
            LogDebug("DigiInfo") << "EE sample " << sample << " ADC counts = " << eeADCCounts[sample] << " Gain Id = " << eeADCGains[sample] << " Analog eq = " << eeAnalogSignal[sample];
          }
        pedestalPreSample /= 3. ; 
        pedestalPreSampleAnalog /= 3. ; 
        double Erec = Emax - pedestalPreSampleAnalog*gainConv_[(int)eeADCGains[Pmax]];
        
        if ( eeSignalSimMap[eeid.rawId()] != 0. ) {
          LogDebug("DigiInfo") << " Digi / Signal Hit = " << Erec << " / " << eeSignalSimMap[eeid.rawId()] << " gainConv " << gainConv_[(int)eeADCGains[Pmax]];
          if ( Erec > 100.*endcapADCtoGeV_  && meEEDigiMixRatiogt100ADC_  ) meEEDigiMixRatiogt100ADC_->Fill( Erec/eeSignalSimMap[eeid.rawId()] );
          if ( eeSignalSimMap[eeid.rawId()] > eeSimThreshold  && meEEDigiMixRatioOriggt40pc_ ) meEEDigiMixRatioOriggt40pc_->Fill( Erec/eeSignalSimMap[eeid.rawId()] );
          if ( eeSignalSimMap[eeid.rawId()] > eeSimThreshold  && meEBShape_ ) {
            for ( int i = 0; i < 10 ; i++ ) {
              meEEShape_->Fill(i, eeAnalogSignal[i] );
            }
          }
        }
      }
    
    isBarrel = false;
    computeCrystalBunchDigi(c, *endcapHits, eeSignalSimMap, isBarrel, eeSimThreshold);
  }

  if ( isPreshower) {

    const std::string preshowerHitsName ("EcalHitsES") ;
    std::auto_ptr<MixCollection<PCaloHit> > 
      preshowerHits (new MixCollection<PCaloHit>(crossingFrame.product (), preshowerHitsName)) ;
    
    for (MixCollection<PCaloHit>::MixItr hitItr = preshowerHits->begin () ;
         hitItr != preshowerHits->end () ;
         ++hitItr) {
      
      ESDetId esid = ESDetId(hitItr->id()) ;
      
      LogDebug("HitInfo") 
        << " CaloHit " << hitItr->getName() << "\n" 
        << " DetID = "<<hitItr->id()<< "ESDetId: z side " << esid.zside() << "  plane " << esid.plane() << esid.six() << ',' << esid.siy() << ':' << esid.strip() << "\n"
        << " Time = " << hitItr->time() << " Event id. = " << hitItr->eventId().rawId() << "\n"
        << " Track Id = " << hitItr->geantTrackId() << "\n"
        << " Energy = " << hitItr->energy();

      if ( meESbunchCrossing_ ) meESbunchCrossing_->Fill(hitItr->eventId().bunchCrossing()); 

    }
    
  }
  
}                                                                                       

void  EcalMixingModuleValidation::checkCalibrations(const edm::EventSetup & eventSetup) 
{

  // ADC -> GeV Scale
  edm::ESHandle<EcalADCToGeVConstant> pAgc;
  eventSetup.get<EcalADCToGeVConstantRcd>().get(pAgc);
  const EcalADCToGeVConstant* agc = pAgc.product();
  
  EcalMGPAGainRatio * defaultRatios = new EcalMGPAGainRatio();

  gainConv_[0] = 0.;
  gainConv_[1] = 1.;
  gainConv_[2] = defaultRatios->gain12Over6() ;
  gainConv_[3] = gainConv_[2]*(defaultRatios->gain6Over1()) ;

  LogDebug("EcalDigi") << " Gains conversions: " << "\n" << " g1 = " << gainConv_[1] << "\n" << " g2 = " << gainConv_[2] << "\n" << " g3 = " << gainConv_[3];

  delete defaultRatios;

  const double barrelADCtoGeV_  = agc->getEBValue();
  LogDebug("EcalDigi") << " Barrel GeV/ADC = " << barrelADCtoGeV_;
  const double endcapADCtoGeV_ = agc->getEEValue();
  LogDebug("EcalDigi") << " Endcap GeV/ADC = " << endcapADCtoGeV_;

}

void EcalMixingModuleValidation::computeCrystalBunchDigi(const edm::EventSetup & eventSetup, MixCollection<PCaloHit> & theHits, MapType & SignalSimMap, const bool & isBarrel, const double & theSimThreshold)
{

  // load the geometry

  edm::ESHandle<CaloGeometry> hGeometry;
  eventSetup.get<IdealGeometryRecord>().get(hGeometry);

  const CaloGeometry * pGeometry = &*hGeometry;
  
  // see if we need to update
  if(pGeometry != theGeometry) {
    theGeometry = pGeometry;
    theEcalResponse->setGeometry(theGeometry);
  }

  // vector of DetId with energy above a fraction of the gun's energy

  std::vector<DetId> theCrystalId;
  if ( isBarrel ) { theCrystalId = theGeometry->getValidDetIds(DetId::Ecal, EcalBarrel); }
  else { theCrystalId = theGeometry->getValidDetIds(DetId::Ecal, EcalEndcap); }

  std::vector<DetId> theOverThresholdId;
  for ( unsigned int i = 0 ; i < theCrystalId.size() ; i++ ) {

    int crysId = theCrystalId[i].rawId();
    if ( SignalSimMap[crysId] > theSimThreshold ) theOverThresholdId.push_back( theCrystalId[i] );

  }

   for (int iBunch = theMinBunch ; iBunch <= theMaxBunch ; iBunch++ ) {

     theEcalResponse->setBunchRange(iBunch, iBunch);
     theEcalResponse->clear();
     theEcalResponse->run(theHits);

     int iHisto = iBunch - theMinBunch;

     for ( std::vector<DetId>::const_iterator idItr = theOverThresholdId.begin() ; idItr != theOverThresholdId.end() ; ++idItr ) {

       CaloSamples * analogSignal = theEcalResponse->findSignal(*idItr);

       if ( analogSignal ) {
        
         (*analogSignal) *= theParameterMap->simParameters(analogSignal->id()).photoelectronsToAnalog();

         for ( int i = 0 ; i < CaloSamples::MAXSAMPLES ; i++ ) {

           if ( isBarrel ) { meEBBunchShape_[iHisto]->Fill(i,(float)(*analogSignal)[i]); }
           else { meEEBunchShape_[iHisto]->Fill(i,(float)(*analogSignal)[i]); }
         }
       }

     }

   }

 }
