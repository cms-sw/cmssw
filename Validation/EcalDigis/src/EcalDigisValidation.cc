/*
 * \file EcalDigisValidation.cc
 *
 * $Date: 2010/01/04 15:10:59 $
 * $Revision: 1.30 $
 * \author F. Cossutti
 *
*/

#include <Validation/EcalDigis/interface/EcalDigisValidation.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include <DataFormats/EcalDetId/interface/ESDetId.h>
#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace cms;
using namespace edm;
using namespace std;

EcalDigisValidation::EcalDigisValidation(const ParameterSet& ps):
  HepMCLabel(ps.getParameter<std::string>("moduleLabelMC")),
  g4InfoLabel(ps.getParameter<std::string>("moduleLabelG4")),
  EBdigiCollection_(ps.getParameter<edm::InputTag>("EBdigiCollection")),
  EEdigiCollection_(ps.getParameter<edm::InputTag>("EEdigiCollection")),
  ESdigiCollection_(ps.getParameter<edm::InputTag>("ESdigiCollection")){

 
  // DQM ROOT output
  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
 
  if ( outputFile_.size() != 0 ) {
    LogInfo("OutputInfo") << " Ecal Digi Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    LogInfo("OutputInfo") << " Ecal Digi Task histograms will NOT be saved";
  }
 
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);
                                                                                                                                           
  dbe_ = 0;
                                                                                                                                          
  // get hold of back-end interface
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

  gainConv_[1] = 1.;
  gainConv_[2] = 2.;
  gainConv_[3] = 12.;
  gainConv_[0] = 12.;   // saturated channels
  barrelADCtoGeV_ = 0.035;
  endcapADCtoGeV_ = 0.06;
 
  meGunEnergy_ = 0;
  meGunEta_ = 0;   
  meGunPhi_ = 0;   

  meEBDigiSimRatio_ = 0;
  meEEDigiSimRatio_ = 0;

  meEBDigiSimRatiogt10ADC_ = 0;
  meEEDigiSimRatiogt20ADC_ = 0;

  meEBDigiSimRatiogt100ADC_ = 0;
  meEEDigiSimRatiogt100ADC_ = 0;

  Char_t histo[200];
 
  
  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalDigisV/EcalDigiTask");
  
    sprintf (histo, "EcalDigiTask Gun Momentum" ) ;
    meGunEnergy_ = dbe_->book1D(histo, histo, 100, 0., 1000.);
  
    sprintf (histo, "EcalDigiTask Gun Eta" ) ;
    meGunEta_ = dbe_->book1D(histo, histo, 700, -3.5, 3.5);
  
    sprintf (histo, "EcalDigiTask Gun Phi" ) ;
    meGunPhi_ = dbe_->book1D(histo, histo, 360, 0., 360.);

    sprintf (histo, "EcalDigiTask Barrel maximum Digi over Sim ratio" ) ;
    meEBDigiSimRatio_ = dbe_->book1D(histo, histo, 100, 0., 2.) ;

    sprintf (histo, "EcalDigiTask Endcap maximum Digi over Sim ratio" ) ;
    meEEDigiSimRatio_ = dbe_->book1D(histo, histo, 100, 0., 2.) ;

    sprintf (histo, "EcalDigiTask Barrel maximum Digi over Sim ratio gt 10 ADC" ) ;
    meEBDigiSimRatiogt10ADC_ = dbe_->book1D(histo, histo, 100, 0., 2.) ;

    sprintf (histo, "EcalDigiTask Endcap maximum Digi over Sim ratio gt 20 ADC" ) ;
    meEEDigiSimRatiogt20ADC_ = dbe_->book1D(histo, histo, 100, 0., 2.) ;

    sprintf (histo, "EcalDigiTask Barrel maximum Digi over Sim ratio gt 100 ADC" ) ;
    meEBDigiSimRatiogt100ADC_ = dbe_->book1D(histo, histo, 100, 0., 2.) ;

    sprintf (histo, "EcalDigiTask Endcap maximum Digi over Sim ratio gt 100 ADC" ) ;
    meEEDigiSimRatiogt100ADC_ = dbe_->book1D(histo, histo, 100, 0., 2.) ;

  }
 
}

EcalDigisValidation::~EcalDigisValidation(){
 
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);

}

void EcalDigisValidation::beginRun(Run const &, EventSetup const & c){

  checkCalibrations(c);

}

void EcalDigisValidation::endJob(){

}

void EcalDigisValidation::analyze(Event const & e, EventSetup const & c){

  LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();

  vector<SimTrack> theSimTracks;
  vector<SimVertex> theSimVertexes;

  Handle<HepMCProduct> MCEvt;
  Handle<SimTrackContainer> SimTk;
  Handle<SimVertexContainer> SimVtx;
  Handle<CrossingFrame<PCaloHit> > crossingFrame;
  Handle<EBDigiCollection> EcalDigiEB;
  Handle<EEDigiCollection> EcalDigiEE;
  Handle<ESDigiCollection> EcalDigiES;
  
  bool skipMC = false;
  e.getByLabel(HepMCLabel, MCEvt);
  if (!MCEvt.isValid()) { skipMC = true; }
  e.getByLabel(g4InfoLabel,SimTk);
  e.getByLabel(g4InfoLabel,SimVtx);

  const EBDigiCollection* EBdigis =0;
  const EEDigiCollection* EEdigis =0;
  const ESDigiCollection* ESdigis =0;

  bool isBarrel = true;
  e.getByLabel( EBdigiCollection_, EcalDigiEB );
  if (EcalDigiEB.isValid()) {
    EBdigis = EcalDigiEB.product();
    LogDebug("DigiInfo") << "total # EBdigis: " << EBdigis->size() ;
    if ( EBdigis->size() == 0 ) isBarrel = false;
  } else {
    isBarrel = false; 
  }
  
  bool isEndcap = true;
  e.getByLabel( EEdigiCollection_, EcalDigiEE );
  if (EcalDigiEE.isValid()) {  
    EEdigis = EcalDigiEE.product();
    LogDebug("DigiInfo") << "total # EEdigis: " << EEdigis->size() ;
    if ( EEdigis->size() == 0 ) isEndcap = false;
  } else {
    isEndcap = false; 
  }
  
  bool isPreshower = true;
  e.getByLabel( ESdigiCollection_, EcalDigiES );
  if (EcalDigiES.isValid()) {
    ESdigis = EcalDigiES.product();
    LogDebug("DigiInfo") << "total # ESdigis: " << ESdigis->size() ;
    if ( ESdigis->size() == 0 ) isPreshower = false;
  } else { 
    isPreshower = false; 
  }

  theSimTracks.insert(theSimTracks.end(),SimTk->begin(),SimTk->end());
  theSimVertexes.insert(theSimVertexes.end(),SimVtx->begin(),SimVtx->end());

  if ( ! skipMC ) {
    double theGunEnergy = 0.;
    for ( HepMC::GenEvent::particle_const_iterator p = MCEvt->GetEvent()->particles_begin();
          p != MCEvt->GetEvent()->particles_end(); ++p ) {
      
      theGunEnergy = (*p)->momentum().e();
      double htheta = (*p)->momentum().theta();
      double heta = -log(tan(htheta * 0.5));
      double hphi = (*p)->momentum().phi();
      hphi = (hphi>=0) ? hphi : hphi+2*M_PI;
      hphi = hphi / M_PI * 180.;
      LogDebug("EventInfo") << "Particle gun type form MC = " << abs((*p)->pdg_id()) << "\n" << "Energy = "<< (*p)->momentum().e() << " Eta = " << heta << " Phi = " << hphi;
      
      if (meGunEnergy_) meGunEnergy_->Fill(theGunEnergy);
      if (meGunEta_) meGunEta_->Fill(heta);
      if (meGunPhi_) meGunPhi_->Fill(hphi);
      
    }
  }
  
  int nvtx = 0;
  for (vector<SimVertex>::iterator isimvtx = theSimVertexes.begin();
       isimvtx != theSimVertexes.end(); ++isimvtx){
    LogDebug("EventInfo") <<" Vertex index = " << nvtx << " event Id = " << isimvtx->eventId().rawId() << "\n" << " vertex dump: " << *isimvtx ;
    ++nvtx;
  }
  
  int ntrk = 0;
  for (vector<SimTrack>::iterator isimtrk = theSimTracks.begin();
       isimtrk != theSimTracks.end(); ++isimtrk){
    LogDebug("EventInfo") <<" Track index = " << ntrk << " track Id = " << isimtrk->trackId() << " event Id = " << isimtrk->eventId().rawId() << "\n" << " track dump: " << *isimtrk ; 
    ++ntrk;
  }
  
  // BARREL

  // loop over simHits

  if ( isBarrel ) {

    const std::string barrelHitsName(g4InfoLabel+"EcalHitsEB");
    e.getByLabel("mix",barrelHitsName,crossingFrame);
    std::auto_ptr<MixCollection<PCaloHit> > 
      barrelHits (new MixCollection<PCaloHit>(crossingFrame.product ()));
    
    MapType ebSimMap;
    
    for (MixCollection<PCaloHit>::MixItr hitItr = barrelHits->begin () ;
         hitItr != barrelHits->end () ;
         ++hitItr) {
      
      EBDetId ebid = EBDetId(hitItr->id()) ;
      
      LogDebug("HitInfo") 
        << " CaloHit "  << hitItr->getName() << "\n" 
        << " DetID = "  << hitItr->id()<< " EBDetId = " << ebid.ieta() << " " << ebid.iphi() << "\n"	
        << " Time = "   << hitItr->time() << " Event id. = " << hitItr->eventId().rawId() << "\n"
        << " Track Id = " << hitItr->geantTrackId() << "\n"
        << " Energy = " << hitItr->energy();

      uint32_t crystid = ebid.rawId();
      ebSimMap[crystid] += hitItr->energy();
      
    }
    
    // loop over Digis
    
    const EBDigiCollection * barrelDigi = EcalDigiEB.product () ;
    
    std::vector<double> ebAnalogSignal ;
    std::vector<double> ebADCCounts ;
    std::vector<double> ebADCGains ;
    ebAnalogSignal.reserve(EBDataFrame::MAXSAMPLES);
    ebADCCounts.reserve(EBDataFrame::MAXSAMPLES);
    ebADCGains.reserve(EBDataFrame::MAXSAMPLES);
    
    for (unsigned int digis=0; digis<EcalDigiEB->size(); ++digis) {
      
      EBDataFrame ebdf=(*barrelDigi)[digis];
      int nrSamples=ebdf.size();
      
      EBDetId ebid = ebdf.id () ;
      
      double Emax = 0. ;
      int Pmax = 0 ;
      double pedestalPreSample = 0.;
      double pedestalPreSampleAnalog = 0.;
      
      for (int sample = 0 ; sample < nrSamples; ++sample) {
	ebAnalogSignal[sample] = 0.;
	ebADCCounts[sample] = 0.;
	ebADCGains[sample] = -1.;
      }
      
      for (int sample = 0 ; sample < nrSamples; ++sample) {
	
	EcalMGPASample mySample = ebdf[sample];
	
	ebADCCounts[sample] = (mySample.adc()) ;
	ebADCGains[sample]  = (mySample.gainId()) ;
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
      
      if ( ebSimMap[ebid.rawId()] != 0. ) {
	LogDebug("DigiInfo") << " Digi / Hit = " << Erec << " / " << ebSimMap[ebid.rawId()] << " gainConv " << gainConv_[(int)ebADCGains[Pmax]];
	if ( meEBDigiSimRatio_ ) meEBDigiSimRatio_->Fill( Erec/ebSimMap[ebid.rawId()] ) ; 
	if ( Erec > 10.*barrelADCtoGeV_  && meEBDigiSimRatiogt10ADC_  ) meEBDigiSimRatiogt10ADC_->Fill( Erec/ebSimMap[ebid.rawId()] );
	if ( Erec > 100.*barrelADCtoGeV_  && meEBDigiSimRatiogt100ADC_  ) meEBDigiSimRatiogt100ADC_->Fill( Erec/ebSimMap[ebid.rawId()] );
	
      }
      
    } 
    
  }
  
  // ENDCAP

  // loop over simHits

  if ( isEndcap ) {

    const std::string endcapHitsName(g4InfoLabel+"EcalHitsEE");
    e.getByLabel("mix",endcapHitsName,crossingFrame);
    std::auto_ptr<MixCollection<PCaloHit> > 
      endcapHits (new MixCollection<PCaloHit>(crossingFrame.product ()));

    MapType eeSimMap;
    
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
      eeSimMap[crystid] += hitItr->energy();

    }
    
    // loop over Digis
    
    const EEDigiCollection * endcapDigi = EcalDigiEE.product () ;
    
    std::vector<double> eeAnalogSignal ;
    std::vector<double> eeADCCounts ;
    std::vector<double> eeADCGains ;
    eeAnalogSignal.reserve(EEDataFrame::MAXSAMPLES);
    eeADCCounts.reserve(EEDataFrame::MAXSAMPLES);
    eeADCGains.reserve(EEDataFrame::MAXSAMPLES);
    
    for (unsigned int digis=0; digis<EcalDigiEE->size(); ++digis) {
      
      EEDataFrame eedf=(*endcapDigi)[digis];
      int nrSamples=eedf.size();
      
      EEDetId eeid = eedf.id () ;
      
      double Emax = 0. ;
      int Pmax = 0 ;
      double pedestalPreSample = 0.;
      double pedestalPreSampleAnalog = 0.;
      
      for (int sample = 0 ; sample < nrSamples; ++sample) {
	eeAnalogSignal[sample] = 0.;
	eeADCCounts[sample] = 0.;
	eeADCGains[sample] = -1.;
      }
      
      for (int sample = 0 ; sample < nrSamples; ++sample) {

	EcalMGPASample mySample = eedf[sample];
	
	eeADCCounts[sample] = (mySample.adc()) ;
	eeADCGains[sample]  = (mySample.gainId()) ;
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
      
      if (eeSimMap[eeid.rawId()] != 0. ) {
	LogDebug("DigiInfo") << " Digi / Hit = " << Erec << " / " << eeSimMap[eeid.rawId()] << " gainConv " << gainConv_[(int)eeADCGains[Pmax]];
	if ( meEEDigiSimRatio_) meEEDigiSimRatio_->Fill( Erec/eeSimMap[eeid.rawId()] ) ; 
	if ( Erec > 20.*endcapADCtoGeV_  && meEEDigiSimRatiogt20ADC_  ) meEEDigiSimRatiogt20ADC_->Fill( Erec/eeSimMap[eeid.rawId()] );
	if ( Erec > 100.*endcapADCtoGeV_  && meEEDigiSimRatiogt100ADC_  ) meEEDigiSimRatiogt100ADC_->Fill( Erec/eeSimMap[eeid.rawId()] );
      }
      
    } 
    
  }

  if ( isPreshower) {

    const std::string preshowerHitsName(g4InfoLabel+"EcalHitsES");
    e.getByLabel("mix",preshowerHitsName,crossingFrame);
    std::auto_ptr<MixCollection<PCaloHit> > 
      preshowerHits (new MixCollection<PCaloHit>(crossingFrame.product ()));
    
    for (MixCollection<PCaloHit>::MixItr hitItr = preshowerHits->begin () ;
         hitItr != preshowerHits->end () ;
         ++hitItr) {
      
      ESDetId esid = ESDetId(hitItr->id()) ;
      
      LogDebug("HitInfo") 
        << " CaloHit " << hitItr->getName() << "\n" 
        << " DetID = " << hitItr->id()<< "ESDetId: z side " << esid.zside() << "  plane " << esid.plane() << esid.six() << ',' << esid.siy() << ':' << esid.strip() << "\n"
        << " Time = "  << hitItr->time() << " Event id. = " << hitItr->eventId().rawId() << "\n"
        << " Track Id = " << hitItr->geantTrackId() << "\n"
        << " Energy = "   << hitItr->energy();

    }
    
  }
  
}                                                                                       

void  EcalDigisValidation::checkCalibrations(edm::EventSetup const & eventSetup) 
{

  // ADC -> GeV Scale
  edm::ESHandle<EcalADCToGeVConstant> pAgc;
  eventSetup.get<EcalADCToGeVConstantRcd>().get(pAgc);
  const EcalADCToGeVConstant* agc = pAgc.product();
  
  EcalMGPAGainRatio * defaultRatios = new EcalMGPAGainRatio();

  gainConv_[1] = 1.;
  gainConv_[2] = defaultRatios->gain12Over6() ;
  gainConv_[3] = gainConv_[2]*(defaultRatios->gain6Over1()) ;
  gainConv_[0] = gainConv_[2]*(defaultRatios->gain6Over1()) ;  // saturated channels

  LogDebug("EcalDigi") << " Gains conversions: " << "\n" << " g1 = " << gainConv_[1] << "\n" << " g2 = " << gainConv_[2] << "\n" << " g3 = " << gainConv_[3];
  LogDebug("EcalDigi") << " Gains conversions: " << "\n" << " saturation = " << gainConv_[0];

  delete defaultRatios;

  const double barrelADCtoGeV_  = agc->getEBValue();
  LogDebug("EcalDigi") << " Barrel GeV/ADC = " << barrelADCtoGeV_;
  const double endcapADCtoGeV_ = agc->getEEValue();
  LogDebug("EcalDigi") << " Endcap GeV/ADC = " << endcapADCtoGeV_;

}
