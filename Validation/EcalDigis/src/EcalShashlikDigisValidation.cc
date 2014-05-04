/*
 * \file EcalShashlikDigisValidation.cc
 *
 * $Date: 2010/01/04 15:10:59 $
 * $Revision: 1.21 $
 * \author F. Cossutti
 *
*/

#include <Validation/EcalDigis/interface/EcalShashlikDigisValidation.h>
#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"
//#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
//#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include "DQMServices/Core/interface/DQMStore.h"

using namespace cms;
using namespace edm;
using namespace std;

EcalShashlikDigisValidation::EcalShashlikDigisValidation(const ParameterSet& ps):
  EKdigiCollection_(ps.getParameter<edm::InputTag>("EKdigiCollection"))
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
  gainConv_[0] = 12.;  // saturated channels
  //  shashlikADCtoGeV_ = 0.06;
 
  meEKDigiOccupancyzp_ = 0;
  meEKDigiOccupancyzm_ = 0;
 
  meEKDigiMultiplicityzp_ = 0;
  meEKDigiMultiplicityzm_ = 0;

  meEKDigiADCGlobal_ = 0;

  for (int i = 0; i < 10 ; i++ ) {
    meEKDigiADCAnalog_[i] = 0;
    meEKDigiADCgS_[i]  = 0;
    meEKDigiADCg1_[i]  = 0;
    meEKDigiADCg6_[i]  = 0;
    meEKDigiADCg12_[i] = 0;
    meEKDigiGain_[i] = 0;
  }

  meEKPedestal_ = 0;
                                 
  meEKMaximumgt100ADC_ = 0; 
                                 
  meEKMaximumgt20ADC_ = 0; 

  meEKnADCafterSwitch_ = 0;
 
  Char_t histo[200];
 
  
  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalDigisV/EcalDigiTask");
  
    sprintf (histo, "EcalDigiTask Shashlik occupancy z+" ) ;
    meEKDigiOccupancyzp_ = dbe_->book2D(histo, histo, 400, 0., 400., 400, 0., 400.);
    
    sprintf (histo, "EcalDigiTask Shashlik occupancy z-" ) ;
    meEKDigiOccupancyzm_ = dbe_->book2D(histo, histo, 400, 0., 400., 400, 0., 400.);
  
    sprintf (histo, "EcalDigiTask Shashlik multiplicity z+" ) ;
    meEKDigiMultiplicityzp_ = dbe_->book1D(histo, histo, 200, 0., 7324.);
    
    sprintf (histo, "EcalDigiTask Shashlik multiplicity z-" ) ;
    meEKDigiMultiplicityzm_ = dbe_->book1D(histo, histo, 200, 0., 7324.);
    
    sprintf (histo, "EcalDigiTask Shashlik global pulse shape" ) ;
    meEKDigiADCGlobal_ = dbe_->bookProfile(histo, histo, 10, 0, 10, 10000, 0., 1000.) ;

    for (int i = 0; i < 10 ; i++ ) {

      sprintf (histo, "EcalDigiTask Shashlik analog pulse %02d", i+1) ;
      meEKDigiADCAnalog_[i] = dbe_->book1D(histo, histo, 4000, 0., 400.);

      sprintf (histo, "EcalDigiTask Shashlik ADC pulse %02d Gain 0 - Saturated", i+1) ;
      meEKDigiADCgS_[i] = dbe_->book1D(histo, histo, 4096, -0.5, 4095.5);

      sprintf (histo, "EcalDigiTask Shashlik ADC pulse %02d Gain 1", i+1) ;
      meEKDigiADCg1_[i] = dbe_->book1D(histo, histo, 4096, -0.5, 4095.5);

      sprintf (histo, "EcalDigiTask Shashlik ADC pulse %02d Gain 6", i+1) ;
      meEKDigiADCg6_[i] = dbe_->book1D(histo, histo, 4096, -0.5, 4095.5);

      sprintf (histo, "EcalDigiTask Shashlik ADC pulse %02d Gain 12", i+1) ;
      meEKDigiADCg12_[i] = dbe_->book1D(histo, histo, 4096, -0.5, 4095.5);

      sprintf (histo, "EcalDigiTask Shashlik gain pulse %02d", i+1) ;
      meEKDigiGain_[i] = dbe_->book1D(histo, histo, 4, 0, 4);
    }
    
    sprintf (histo, "EcalDigiTask Shashlik pedestal for pre-sample" ) ;
    meEKPedestal_ = dbe_->book1D(histo, histo, 4096, -0.5, 4095.5) ;

    sprintf (histo, "EcalDigiTask Shashlik maximum position gt 100 ADC" ) ;
    meEKMaximumgt100ADC_ = dbe_->book1D(histo, histo, 10, 0., 10.) ;

    sprintf (histo, "EcalDigiTask Shashlik maximum position gt 20 ADC" ) ;
    meEKMaximumgt20ADC_ = dbe_->book1D(histo, histo, 10, 0., 10.) ;

    sprintf (histo, "EcalDigiTask Shashlik ADC counts after gain switch" ) ;
    meEKnADCafterSwitch_ = dbe_->book1D(histo, histo, 10, 0., 10.) ;

  }
 
}

EcalShashlikDigisValidation::~EcalShashlikDigisValidation(){
 
}

void EcalShashlikDigisValidation::beginRun(Run const &, EventSetup const & c){

  checkCalibrations(c);

}

void EcalShashlikDigisValidation::endJob(){

}

void EcalShashlikDigisValidation::analyze(Event const & e, EventSetup const & eventSetup){

  //LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();

  Handle<EKDigiCollection> EcalDigiEK;

  e.getByLabel( EKdigiCollection_ , EcalDigiEK );

  // ADC -> GeV Scale
  edm::ESHandle<EcalADCToGeVConstant> pAgc;
  eventSetup.get<EcalADCToGeVConstantRcd>().get(pAgc);
  const EcalADCToGeVConstant* adctogev = pAgc.product();

  // Return if no Shashlik data available
  if( !EcalDigiEK.isValid() ) return;

  // SHASHLIK

  // loop over Digis

  const EKDigiCollection * shashlikDigi = EcalDigiEK.product () ;

  std::vector<double> eeAnalogSignal ;
  std::vector<double> eeADCCounts ;
  std::vector<double> eeADCGains ;
  eeAnalogSignal.reserve(EKDataFrame::MAXSAMPLES);
  eeADCCounts.reserve(EKDataFrame::MAXSAMPLES);
  eeADCGains.reserve(EKDataFrame::MAXSAMPLES);

  int nDigiszp = 0;
  int nDigiszm = 0;

  for (unsigned int digis=0; digis<EcalDigiEK->size(); ++digis) {
    
    EKDataFrame eedf=(*shashlikDigi)[digis];
    int nrSamples=eedf.size();
    
    EKDetId eeid = eedf.id () ;
    
    if (eeid.zside() > 0 ) {
      if (meEKDigiOccupancyzp_) meEKDigiOccupancyzp_->Fill( eeid.ix(), eeid.iy() );
      nDigiszp++;
    }
    else if (eeid.zside() < 0 ) {
      if (meEKDigiOccupancyzm_) meEKDigiOccupancyzm_->Fill( eeid.ix(), eeid.iy() );
      nDigiszm++;
    }
    
    double Emax = 0. ;
    int Pmax = 0 ;
    double pedestalPreSample = 0.;
    double pedestalPreSampleAnalog = 0.;
    int countsAfterGainSwitch = -1;
    double higherGain = 1.;
    int higherGainSample = 0;
    
    for (int sample = 0 ; sample < nrSamples; ++sample) {
      eeAnalogSignal[sample] = 0.;
      eeADCCounts[sample] = 0.;
      eeADCGains[sample] = 0.;
    }
    
    for (int sample = 0 ; sample < nrSamples; ++sample) {
      
      EcalMGPASample mySample = eedf[sample];
      
      eeADCCounts[sample] = (mySample.adc());
      eeADCGains[sample]  = (mySample.gainId()) ;
      eeAnalogSignal[sample] = (eeADCCounts[sample]*gainConv_[(int)eeADCGains[sample]]*  adctogev->getEKValue());
      
      if (Emax < eeAnalogSignal[sample] ) {
	Emax = eeAnalogSignal[sample] ;
	Pmax = sample ;
      }
      
      if ( sample < 3 ) {
	pedestalPreSample += eeADCCounts[sample] ;
	pedestalPreSampleAnalog += eeADCCounts[sample]*gainConv_[(int)eeADCGains[sample]]*shashlikADCtoGeV_ ;
      }

      if (sample > 0 && ( ((eeADCGains[sample] > eeADCGains[sample-1]) && (eeADCGains[sample-1]!=0)) || (countsAfterGainSwitch<0 && eeADCGains[sample]==0)) ){  
	higherGain = eeADCGains[sample];
	higherGainSample = sample;
	countsAfterGainSwitch = 1;
      }
      
      if ( (higherGain > 1 && (higherGainSample != sample) && (eeADCGains[sample] == higherGain)) || (higherGain==3 && (higherGainSample != sample) && (eeADCGains[sample]==0)) || (higherGain==0 && (higherGainSample != sample) && ((eeADCGains[sample]==0) || (eeADCGains[sample]==3))) ) countsAfterGainSwitch++ ;
    }
    pedestalPreSample /= 3. ; 
    pedestalPreSampleAnalog /= 3. ; 
    
    LogDebug("DigiInfo") << "Shashlik Digi for EKDetId = " << eeid.rawId() << " x,y " << eeid.ix() << " " << eeid.iy() ;
    for ( int i = 0; i < 10 ; i++ ) {
      LogDebug("DigiInfo") << "sample " << i << " ADC = " << eeADCCounts[i] << " gain = " << eeADCGains[i] << " Analog = " << eeAnalogSignal[i] ;
      //std::cout << "[DigiInfo]" << "sample " << i << " ADC = " << eeADCCounts[i] << " gain = " << eeADCGains[i] << " Analog = " << eeAnalogSignal[i] << std::endl;

    }
    LogDebug("DigiInfo") << "Maximum energy = " << Emax << " in sample " << Pmax << " Pedestal from pre-sample = " << pedestalPreSampleAnalog;  
    if(Emax>13) std::cout << "[DigiInfo]" << "Maximum energy = " << Emax << " in sample " << Pmax << " Pedestal from pre-sample = " << pedestalPreSampleAnalog 
			  << "\tdetid = " << eeid.rawId() << " ix= " << eeid.ix() << " iy= " << eeid.iy()
			  << std::endl;
    if ( countsAfterGainSwitch > 0 ) LogDebug("DigiInfo") << "Counts after switch " << countsAfterGainSwitch; 
    
    if ( countsAfterGainSwitch > 0 && countsAfterGainSwitch < 5 ) {
      edm::LogWarning("DigiWarning") << "Wrong number of counts after gain switch before next switch! " << countsAfterGainSwitch ;
      for ( int i = 0; i < 10 ; i++ ) {
	edm::LogWarning("DigiWarning") << "sample " << i << " ADC = " << eeADCCounts[i] << " gain = " << eeADCGains[i] << " Analog = " << eeAnalogSignal[i];
      }
    }
    
    for ( int i = 0 ; i < 10 ; i++ ) {
      if (meEKDigiADCGlobal_ && (Emax-pedestalPreSampleAnalog*gainConv_[(int)eeADCGains[Pmax]]) > 100.*shashlikADCtoGeV_) meEKDigiADCGlobal_->Fill( i , eeAnalogSignal[i] ) ;
      if (meEKDigiADCAnalog_[i]) meEKDigiADCAnalog_[i]->Fill( eeAnalogSignal[i] ) ;
      if ( eeADCGains[i] == 0 ) {
	if (meEKDigiADCgS_[i]) meEKDigiADCgS_[i]->Fill( eeADCCounts[i] ) ;
      }
      else if ( eeADCGains[i] == 3 ) {
	if (meEKDigiADCg1_[i]) meEKDigiADCg1_[i]->Fill( eeADCCounts[i] ) ;
      }
      else if ( eeADCGains[i] == 2 ) {
	if (meEKDigiADCg6_[i]) meEKDigiADCg6_[i]->Fill( eeADCCounts[i] ) ;
      }
      else if ( eeADCGains[i] == 1 ) {
	if (meEKDigiADCg12_[i]) meEKDigiADCg12_[i]->Fill( eeADCCounts[i] ) ;
      }
      if (meEKDigiGain_[i]) meEKDigiGain_[i]->Fill( eeADCGains[i] ) ;
    }
    
    if (meEKPedestal_) meEKPedestal_->Fill ( pedestalPreSample ) ;
    if (meEKMaximumgt20ADC_ && (Emax-pedestalPreSampleAnalog*gainConv_[(int)eeADCGains[Pmax]]) > 20.*shashlikADCtoGeV_) meEKMaximumgt20ADC_->Fill( Pmax ) ;
    if (meEKMaximumgt100ADC_ && (Emax-pedestalPreSampleAnalog*gainConv_[(int)eeADCGains[Pmax]]) > 100.*shashlikADCtoGeV_) meEKMaximumgt100ADC_->Fill( Pmax ) ;
    if (meEKnADCafterSwitch_) meEKnADCafterSwitch_->Fill(countsAfterGainSwitch);
    
  } 
  
  if ( meEKDigiMultiplicityzp_ ) meEKDigiMultiplicityzp_->Fill(nDigiszp);
  if ( meEKDigiMultiplicityzm_ ) meEKDigiMultiplicityzm_->Fill(nDigiszm);
  
}

void  EcalShashlikDigisValidation::checkCalibrations(edm::EventSetup const & eventSetup) 
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

  LogDebug("EcalDigi") << " Gains conversions: " << "\n" << " g0 = " << gainConv_[0] << "\n" << " g1 = " << gainConv_[1] << "\n" << " g2 = " << gainConv_[2] << "\n" << " g3 = " << gainConv_[3]; 

  delete defaultRatios;

  const double shashlikADCtoGeV_ = agc->getEKValue();
  LogDebug("EcalDigi") << " Shashlik GeV/ADC = " << shashlikADCtoGeV_;

}

                                                                                                                                                             
DEFINE_FWK_MODULE(EcalShashlikDigisValidation);
