/*
 * \file EcalEndcapDigisValidation.cc
 *
 * \author F. Cossutti
 *
*/

#include <Validation/EcalDigis/interface/EcalEndcapDigisValidation.h>
#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"

using namespace cms;
using namespace edm;
using namespace std;

EcalEndcapDigisValidation::EcalEndcapDigisValidation(const ParameterSet& ps):
  EEdigiCollectionToken_(consumes<EEDigiCollection>(ps.getParameter<edm::InputTag>("EEdigiCollection")))
  {
 
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  gainConv_[1] = 1.;
  gainConv_[2] = 2.;
  gainConv_[3] = 12.;
  gainConv_[0] = 12.;  // saturated channels
  barrelADCtoGeV_ = 0.035;
  endcapADCtoGeV_ = 0.06;
 
  meEEDigiOccupancyzp_ = 0;
  meEEDigiOccupancyzm_ = 0;
 
  meEEDigiMultiplicityzp_ = 0;
  meEEDigiMultiplicityzm_ = 0;

  meEEDigiADCGlobal_ = 0;

  for (int i = 0; i < 10 ; i++ ) {
    meEEDigiADCAnalog_[i] = 0;
    meEEDigiADCgS_[i]  = 0;
    meEEDigiADCg1_[i]  = 0;
    meEEDigiADCg6_[i]  = 0;
    meEEDigiADCg12_[i] = 0;
    meEEDigiGain_[i] = 0;
  }

  meEEPedestal_ = 0;
                                 
  meEEMaximumgt100ADC_ = 0; 
                                 
  meEEMaximumgt20ADC_ = 0; 

  meEEnADCafterSwitch_ = 0;
 
}

EcalEndcapDigisValidation::~EcalEndcapDigisValidation(){
 
}

void EcalEndcapDigisValidation::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const&, edm::EventSetup const&){

    Char_t histo[200];

    ibooker.setCurrentFolder("EcalDigisV/EcalDigiTask");
  
    sprintf (histo, "EcalDigiTask Endcap occupancy z+" ) ;
    meEEDigiOccupancyzp_ = ibooker.book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    
    sprintf (histo, "EcalDigiTask Endcap occupancy z-" ) ;
    meEEDigiOccupancyzm_ = ibooker.book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
  
    sprintf (histo, "EcalDigiTask Endcap multiplicity z+" ) ;
    meEEDigiMultiplicityzp_ = ibooker.book1D(histo, histo, 100, 0., 7324.);
    
    sprintf (histo, "EcalDigiTask Endcap multiplicity z-" ) ;
    meEEDigiMultiplicityzm_ = ibooker.book1D(histo, histo, 100, 0., 7324.);
    
    sprintf (histo, "EcalDigiTask Endcap global pulse shape" ) ;
    meEEDigiADCGlobal_ = ibooker.bookProfile(histo, histo, 10, 0, 10, 10000, 0., 1000.) ;

    for (int i = 0; i < 10 ; i++ ) {

      sprintf (histo, "EcalDigiTask Endcap analog pulse %02d", i+1) ;
      meEEDigiADCAnalog_[i] = ibooker.book1D(histo, histo, 4000, 0., 400.);

      sprintf (histo, "EcalDigiTask Endcap ADC pulse %02d Gain 0 - Saturated", i+1) ;
      meEEDigiADCgS_[i] = ibooker.book1D(histo, histo, 4096, -0.5, 4095.5);

      sprintf (histo, "EcalDigiTask Endcap ADC pulse %02d Gain 1", i+1) ;
      meEEDigiADCg1_[i] = ibooker.book1D(histo, histo, 4096, -0.5, 4095.5);

      sprintf (histo, "EcalDigiTask Endcap ADC pulse %02d Gain 6", i+1) ;
      meEEDigiADCg6_[i] = ibooker.book1D(histo, histo, 4096, -0.5, 4095.5);

      sprintf (histo, "EcalDigiTask Endcap ADC pulse %02d Gain 12", i+1) ;
      meEEDigiADCg12_[i] = ibooker.book1D(histo, histo, 4096, -0.5, 4095.5);

      sprintf (histo, "EcalDigiTask Endcap gain pulse %02d", i+1) ;
      meEEDigiGain_[i] = ibooker.book1D(histo, histo, 4, 0, 4);
    }
    
    sprintf (histo, "EcalDigiTask Endcap pedestal for pre-sample" ) ;
    meEEPedestal_ = ibooker.book1D(histo, histo, 4096, -0.5, 4095.5) ;

    sprintf (histo, "EcalDigiTask Endcap maximum position gt 100 ADC" ) ;
    meEEMaximumgt100ADC_ = ibooker.book1D(histo, histo, 10, 0., 10.) ;

    sprintf (histo, "EcalDigiTask Endcap maximum position gt 20 ADC" ) ;
    meEEMaximumgt20ADC_ = ibooker.book1D(histo, histo, 10, 0., 10.) ;

    sprintf (histo, "EcalDigiTask Endcap ADC counts after gain switch" ) ;
    meEEnADCafterSwitch_ = ibooker.book1D(histo, histo, 10, 0., 10.) ;

}

void EcalEndcapDigisValidation::dqmBeginRun(edm::Run const&, edm::EventSetup const& c){

  checkCalibrations(c);

} 


void EcalEndcapDigisValidation::analyze(Event const & e, EventSetup const & c){

  //LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();

  Handle<EEDigiCollection> EcalDigiEE;

  e.getByToken( EEdigiCollectionToken_ , EcalDigiEE );

  // Return if no Endcap data available
  if( !EcalDigiEE.isValid() ) return;

  // ENDCAP

  // loop over Digis

  const EEDigiCollection * endcapDigi = EcalDigiEE.product () ;

  std::vector<double> eeAnalogSignal ;
  std::vector<double> eeADCCounts ;
  std::vector<double> eeADCGains ;
  eeAnalogSignal.reserve(EEDataFrame::MAXSAMPLES);
  eeADCCounts.reserve(EEDataFrame::MAXSAMPLES);
  eeADCGains.reserve(EEDataFrame::MAXSAMPLES);

  int nDigiszp = 0;
  int nDigiszm = 0;

  for (unsigned int digis=0; digis<EcalDigiEE->size(); ++digis) {
    
    EEDataFrame eedf=(*endcapDigi)[digis];
    int nrSamples=eedf.size();
    
    EEDetId eeid = eedf.id () ;
    
    if (eeid.zside() > 0 ) {
      if (meEEDigiOccupancyzp_) meEEDigiOccupancyzp_->Fill( eeid.ix(), eeid.iy() );
      nDigiszp++;
    }
    else if (eeid.zside() < 0 ) {
      if (meEEDigiOccupancyzm_) meEEDigiOccupancyzm_->Fill( eeid.ix(), eeid.iy() );
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
      eeAnalogSignal[sample] = (eeADCCounts[sample]*gainConv_[(int)eeADCGains[sample]]*endcapADCtoGeV_);
      
      if (Emax < eeAnalogSignal[sample] ) {
	Emax = eeAnalogSignal[sample] ;
	Pmax = sample ;
      }
      
      if ( sample < 3 ) {
	pedestalPreSample += eeADCCounts[sample] ;
	pedestalPreSampleAnalog += eeADCCounts[sample]*gainConv_[(int)eeADCGains[sample]]*endcapADCtoGeV_ ;
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
    
    LogDebug("DigiInfo") << "Endcap Digi for EEDetId = " << eeid.rawId() << " x,y " << eeid.ix() << " " << eeid.iy() ;
    for ( int i = 0; i < 10 ; i++ ) {
      LogDebug("DigiInfo") << "sample " << i << " ADC = " << eeADCCounts[i] << " gain = " << eeADCGains[i] << " Analog = " << eeAnalogSignal[i] ;
    }
    LogDebug("DigiInfo") << "Maximum energy = " << Emax << " in sample " << Pmax << " Pedestal from pre-sample = " << pedestalPreSampleAnalog;  
    if ( countsAfterGainSwitch > 0 ) LogDebug("DigiInfo") << "Counts after switch " << countsAfterGainSwitch; 
    
    if ( countsAfterGainSwitch > 0 && countsAfterGainSwitch < 5 ) {
      edm::LogWarning("DigiWarning") << "Wrong number of counts after gain switch before next switch! " << countsAfterGainSwitch ;
      for ( int i = 0; i < 10 ; i++ ) {
	edm::LogWarning("DigiWarning") << "sample " << i << " ADC = " << eeADCCounts[i] << " gain = " << eeADCGains[i] << " Analog = " << eeAnalogSignal[i];
      }
    }
    
    for ( int i = 0 ; i < 10 ; i++ ) {
      if (meEEDigiADCGlobal_ && (Emax-pedestalPreSampleAnalog*gainConv_[(int)eeADCGains[Pmax]]) > 100.*endcapADCtoGeV_) meEEDigiADCGlobal_->Fill( i , eeAnalogSignal[i] ) ;
      if (meEEDigiADCAnalog_[i]) meEEDigiADCAnalog_[i]->Fill( eeAnalogSignal[i] ) ;
      if ( eeADCGains[i] == 0 ) {
	if (meEEDigiADCgS_[i]) meEEDigiADCgS_[i]->Fill( eeADCCounts[i] ) ;
      }
      else if ( eeADCGains[i] == 3 ) {
	if (meEEDigiADCg1_[i]) meEEDigiADCg1_[i]->Fill( eeADCCounts[i] ) ;
      }
      else if ( eeADCGains[i] == 2 ) {
	if (meEEDigiADCg6_[i]) meEEDigiADCg6_[i]->Fill( eeADCCounts[i] ) ;
      }
      else if ( eeADCGains[i] == 1 ) {
	if (meEEDigiADCg12_[i]) meEEDigiADCg12_[i]->Fill( eeADCCounts[i] ) ;
      }
      if (meEEDigiGain_[i]) meEEDigiGain_[i]->Fill( eeADCGains[i] ) ;
    }
    
    if (meEEPedestal_) meEEPedestal_->Fill ( pedestalPreSample ) ;
    if (meEEMaximumgt20ADC_ && (Emax-pedestalPreSampleAnalog*gainConv_[(int)eeADCGains[Pmax]]) > 20.*endcapADCtoGeV_) meEEMaximumgt20ADC_->Fill( Pmax ) ;
    if (meEEMaximumgt100ADC_ && (Emax-pedestalPreSampleAnalog*gainConv_[(int)eeADCGains[Pmax]]) > 100.*endcapADCtoGeV_) meEEMaximumgt100ADC_->Fill( Pmax ) ;
    if (meEEnADCafterSwitch_) meEEnADCafterSwitch_->Fill(countsAfterGainSwitch);
    
  } 
  
  if ( meEEDigiMultiplicityzp_ ) meEEDigiMultiplicityzp_->Fill(nDigiszp);
  if ( meEEDigiMultiplicityzm_ ) meEEDigiMultiplicityzm_->Fill(nDigiszm);
  
}

void  EcalEndcapDigisValidation::checkCalibrations(edm::EventSetup const & eventSetup) 
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

  const double barrelADCtoGeV_  = agc->getEBValue();
  LogDebug("EcalDigi") << " Barrel GeV/ADC = " << barrelADCtoGeV_;
  const double endcapADCtoGeV_ = agc->getEEValue();
  LogDebug("EcalDigi") << " Endcap GeV/ADC = " << endcapADCtoGeV_;

}

                                                                                                                                                             
