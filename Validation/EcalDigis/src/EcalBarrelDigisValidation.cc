/*
 * \file EcalBarrelDigisValidation.cc
 *
 * \author F. Cossutti
 *
*/

#include <Validation/EcalDigis/interface/EcalBarrelDigisValidation.h>
#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace cms;
using namespace edm;
using namespace std;

EcalBarrelDigisValidation::EcalBarrelDigisValidation(const ParameterSet& ps):
  EBdigiCollection_(consumes<EBDigiCollection>(ps.getParameter<edm::InputTag>("EBdigiCollection")))
{
  
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
 
  meEBDigiOccupancy_ = 0;

  meEBDigiMultiplicity_ = 0;

  meEBDigiADCGlobal_ = 0;

  for (int i = 0; i < 10 ; i++ ) {
    meEBDigiADCAnalog_[i] = 0;
    meEBDigiADCgS_[i]  = 0;
    meEBDigiADCg1_[i]  = 0;
    meEBDigiADCg6_[i]  = 0;
    meEBDigiADCg12_[i] = 0;
    meEBDigiGain_[i] = 0;
  }

  meEBPedestal_ = 0;
                                 
  meEBMaximumgt100ADC_ = 0; 
                                 
  meEBMaximumgt10ADC_ = 0; 

  meEBnADCafterSwitch_ = 0;
 
  Char_t histo[200];
 
  
  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalDigisV/EcalDigiTask");
  
    sprintf (histo, "EcalDigiTask Barrel occupancy" ) ;
    meEBDigiOccupancy_ = dbe_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);

    sprintf (histo, "EcalDigiTask Barrel digis multiplicity" ) ;
    meEBDigiMultiplicity_ = dbe_->book1D(histo, histo, 612, 0., 61200);
  
    sprintf (histo, "EcalDigiTask Barrel global pulse shape" ) ;
    meEBDigiADCGlobal_ = dbe_->bookProfile(histo, histo, 10, 0, 10, 10000, 0., 1000.) ;
    
    for (int i = 0; i < 10 ; i++ ) {

      sprintf (histo, "EcalDigiTask Barrel analog pulse %02d", i+1) ;
      meEBDigiADCAnalog_[i] = dbe_->book1D(histo, histo, 4000, 0., 400.);

      sprintf (histo, "EcalDigiTask Barrel ADC pulse %02d Gain 0 - Saturated", i+1) ;
      meEBDigiADCgS_[i] = dbe_->book1D(histo, histo, 4096, -0.5, 4095.5);

      sprintf (histo, "EcalDigiTask Barrel ADC pulse %02d Gain 1", i+1) ;
      meEBDigiADCg1_[i] = dbe_->book1D(histo, histo, 4096, -0.5, 4095.5);

      sprintf (histo, "EcalDigiTask Barrel ADC pulse %02d Gain 6", i+1) ;
      meEBDigiADCg6_[i] = dbe_->book1D(histo, histo, 4096, -0.5, 4095.5);

      sprintf (histo, "EcalDigiTask Barrel ADC pulse %02d Gain 12", i+1) ;
      meEBDigiADCg12_[i] = dbe_->book1D(histo, histo, 4096, -0.5, 4095.5);

      sprintf (histo, "EcalDigiTask Barrel gain pulse %02d", i+1) ;
      meEBDigiGain_[i] = dbe_->book1D(histo, histo, 4, 0, 4);

    }
    
    sprintf (histo, "EcalDigiTask Barrel pedestal for pre-sample" ) ;
    meEBPedestal_ = dbe_->book1D(histo, histo, 4096, -0.5, 4095.5) ;

    sprintf (histo, "EcalDigiTask Barrel maximum position gt 100 ADC" ) ;
    meEBMaximumgt100ADC_ = dbe_->book1D(histo, histo, 10, 0., 10.) ;

    sprintf (histo, "EcalDigiTask Barrel maximum position gt 10 ADC" ) ;
    meEBMaximumgt10ADC_ = dbe_->book1D(histo, histo, 10, 0., 10.) ;

    sprintf (histo, "EcalDigiTask Barrel ADC counts after gain switch" ) ;
    meEBnADCafterSwitch_ = dbe_->book1D(histo, histo, 10, 0., 10.) ;

  }
 
}

EcalBarrelDigisValidation::~EcalBarrelDigisValidation(){
 
}

void EcalBarrelDigisValidation::beginRun(Run const &, EventSetup const & c){

  checkCalibrations(c);

}

void EcalBarrelDigisValidation::endJob(){

}

void EcalBarrelDigisValidation::analyze(Event const & e, EventSetup const & c){

  //LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();

  Handle<EBDigiCollection> EcalDigiEB;

  e.getByToken( EBdigiCollection_ , EcalDigiEB );

  //Return if no Barrel data 
  if( !EcalDigiEB.isValid() ) return;

  // BARREL

  // loop over Digis

  const EBDigiCollection * barrelDigi = EcalDigiEB.product () ;

  std::vector<double> ebAnalogSignal ;
  std::vector<double> ebADCCounts ;
  std::vector<double> ebADCGains ;
  ebAnalogSignal.reserve(EBDataFrame::MAXSAMPLES);
  ebADCCounts.reserve(EBDataFrame::MAXSAMPLES);
  ebADCGains.reserve(EBDataFrame::MAXSAMPLES);

  int nDigis = 0;
  
  for (unsigned int digis=0; digis<EcalDigiEB->size(); ++digis) 
    {
      
      EBDataFrame ebdf = (*barrelDigi)[digis];
      int nrSamples = ebdf.size();
      
      EBDetId ebid = ebdf.id () ;

      nDigis++;
      if (meEBDigiOccupancy_) meEBDigiOccupancy_->Fill( ebid.iphi(), ebid.ieta() );
      
      double Emax = 0. ;
      int Pmax = 0 ;
      double pedestalPreSample = 0.;
      double pedestalPreSampleAnalog = 0.;
      int countsAfterGainSwitch = -1;
      double higherGain = 1.;
      int higherGainSample = 0;
      
      for (int sample = 0 ; sample < nrSamples; ++sample) {
        ebAnalogSignal[sample] = 0.;
        ebADCCounts[sample] = 0.;
        ebADCGains[sample] = 0.;
      }
      
      for (int sample = 0 ; sample < nrSamples; ++sample) 
        {
	  EcalMGPASample thisSample = ebdf[sample];
	  
          ebADCCounts[sample] = (thisSample.adc());
          ebADCGains[sample]  = (thisSample.gainId());
          ebAnalogSignal[sample] = (ebADCCounts[sample]*gainConv_[(int)ebADCGains[sample]]*barrelADCtoGeV_);
	  
          if (Emax < ebAnalogSignal[sample] ) {
            Emax = ebAnalogSignal[sample] ;
            Pmax = sample ;
          }
	  
          if ( sample < 3 ) {
            pedestalPreSample += ebADCCounts[sample] ;
            pedestalPreSampleAnalog += ebADCCounts[sample]*gainConv_[(int)ebADCGains[sample]]*barrelADCtoGeV_ ;
          }
	  
          if ( sample > 0 && ( ((ebADCGains[sample] > ebADCGains[sample-1]) && (ebADCGains[sample-1]!=0)) || (countsAfterGainSwitch<0 && ebADCGains[sample]==0)) ) {
            higherGain = ebADCGains[sample];
            higherGainSample = sample;
            countsAfterGainSwitch = 1;
          }
	  
          if ( (higherGain > 1 && (higherGainSample != sample) && (ebADCGains[sample] == higherGain)) || (higherGain==3 && (higherGainSample != sample) && (ebADCGains[sample]==0)) || (higherGain==0 && (higherGainSample != sample) && ((ebADCGains[sample] == 3) || (ebADCGains[sample]==0))) ) { countsAfterGainSwitch++ ; }
        }
      
      pedestalPreSample /= 3. ; 
      pedestalPreSampleAnalog /= 3. ; 
      
      LogDebug("DigiInfo") << "Barrel Digi for EBDetId = " << ebid.rawId() << " eta,phi " << ebid.ieta() << " " << ebid.iphi() ;
      for ( int i = 0; i < 10 ; i++ ) {
	LogDebug("DigiInfo") << "sample " << i << " ADC = " << ebADCCounts[i] << " gain = " << ebADCGains[i] << " Analog = " << ebAnalogSignal[i];  
      }
      LogDebug("DigiInfo") << "Maximum energy = " << Emax << " in sample " << Pmax << " Pedestal from pre-sample = " << pedestalPreSampleAnalog;
      if ( countsAfterGainSwitch > 0 ) LogDebug("DigiInfo") << "Counts after switch " << countsAfterGainSwitch;
      
      if ( countsAfterGainSwitch > 0 && countsAfterGainSwitch < 5 ) {
	edm::LogWarning("DigiWarning") << "Wrong number of counts after gain switch before next switch! " << countsAfterGainSwitch ;
	for ( int i = 0; i < 10 ; i++ ) {
	  edm::LogWarning("DigiWarning") << "sample " << i << " ADC = " << ebADCCounts[i] << " gain = " << ebADCGains[i] << " Analog = " << ebAnalogSignal[i];
	} 
      }
      
      for ( int i = 0 ; i < 10 ; i++ ) {
	if (meEBDigiADCGlobal_ && (Emax-pedestalPreSampleAnalog*gainConv_[(int)ebADCGains[Pmax]]) > 100.*barrelADCtoGeV_) meEBDigiADCGlobal_->Fill( i , ebAnalogSignal[i] ) ;
	if (meEBDigiADCAnalog_[i]) meEBDigiADCAnalog_[i]->Fill( ebAnalogSignal[i] ) ;
	
	if ( ebADCGains[i] == 0) {
	  if (meEBDigiADCgS_[i]) meEBDigiADCgS_[i]->Fill( ebADCCounts[i] ) ;
	}
	else if ( ebADCGains[i] == 3 ) {
	  if (meEBDigiADCg1_[i]) meEBDigiADCg1_[i]->Fill( ebADCCounts[i] ) ;
	}
	else if ( ebADCGains[i] == 2 ) {
	  if (meEBDigiADCg6_[i]) meEBDigiADCg6_[i]->Fill( ebADCCounts[i] ) ;
	}
	else if ( ebADCGains[i] == 1 ) {
	  if (meEBDigiADCg12_[i]) meEBDigiADCg12_[i]->Fill( ebADCCounts[i] ) ;
	}
	if (meEBDigiGain_[i]) meEBDigiGain_[i]->Fill( ebADCGains[i] ) ;
      }
      
      if (meEBPedestal_) meEBPedestal_->Fill ( pedestalPreSample ) ;
      if (meEBMaximumgt10ADC_ && (Emax-pedestalPreSampleAnalog*gainConv_[(int)ebADCGains[Pmax]]) > 10.*barrelADCtoGeV_) meEBMaximumgt10ADC_->Fill( Pmax ) ;
      if (meEBMaximumgt100ADC_ && (Emax-pedestalPreSampleAnalog*gainConv_[(int)ebADCGains[Pmax]]) > 100.*barrelADCtoGeV_) meEBMaximumgt100ADC_->Fill( Pmax ) ;
      if (meEBnADCafterSwitch_) meEBnADCafterSwitch_->Fill( countsAfterGainSwitch ) ;
      
    }
  
  if ( meEBDigiMultiplicity_ ) meEBDigiMultiplicity_->Fill(nDigis);
    
}

void  EcalBarrelDigisValidation::checkCalibrations(edm::EventSetup const & eventSetup) 
  {
    
  // ADC -> GeV Scale
  edm::ESHandle<EcalADCToGeVConstant> pAgc;
  eventSetup.get<EcalADCToGeVConstantRcd>().get(pAgc);
  const EcalADCToGeVConstant* agc = pAgc.product();
  
  EcalMGPAGainRatio * defaultRatios = new EcalMGPAGainRatio();

  gainConv_[1] = 1.;
  gainConv_[2] = defaultRatios->gain12Over6() ;
  gainConv_[3] = gainConv_[2]*(defaultRatios->gain6Over1()) ;
  gainConv_[0] = gainConv_[2]*(defaultRatios->gain6Over1()) ;

  LogDebug("EcalDigi") << " Gains conversions: " << "\n" << " g0 = " << gainConv_[0] << "\n" << " g1 = " << gainConv_[1] << "\n" << " g2 = " << gainConv_[2]  << "\n" << " g3 = " << gainConv_[3]; 

  delete defaultRatios;

  const double barrelADCtoGeV_  = agc->getEBValue();
  LogDebug("EcalDigi") << " Barrel GeV/ADC = " << barrelADCtoGeV_;
  const double endcapADCtoGeV_ = agc->getEEValue();
  LogDebug("EcalDigi") << " Endcap GeV/ADC = " << endcapADCtoGeV_;

}

                                                                                                                                                             
