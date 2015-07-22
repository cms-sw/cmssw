/*
 * \file EcalPreshowerDigisValidation.cc
 *
 * \author F. Cossutti
 *
*/

#include <Validation/EcalDigis/interface/EcalPreshowerDigisValidation.h>

EcalPreshowerDigisValidation::EcalPreshowerDigisValidation(const edm::ParameterSet& ps):
  ESdigiCollectionToken_( consumes<ESDigiCollection>( ps.getParameter<edm::InputTag>( "ESdigiCollection" ) ) )
{
  
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);
                                                                                                                                           
  meESDigiMultiplicity_=0;

  for (int i = 0; i < 3 ; i++ ) {
    meESDigiADC_[i] = 0;
  }
 
}

void EcalPreshowerDigisValidation::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const&, edm::EventSetup const&){

  Char_t histo[200];
 
  ibooker.setCurrentFolder("EcalDigisV/EcalDigiTask");

  sprintf (histo, "EcalDigiTask Preshower digis multiplicity" ) ;
  meESDigiMultiplicity_ = ibooker.book1D(histo, histo, 1000, 0., 137728);

  for ( int i = 0; i < 3 ; i++ ) {

    sprintf (histo, "EcalDigiTask Preshower ADC pulse %02d", i+1) ; 
      meESDigiADC_[i] = ibooker.book1D(histo, histo, 4096, -0.5, 4095.5) ;
  }
}

void EcalPreshowerDigisValidation::analyze(const edm::Event& e, const edm::EventSetup& c){

  //LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();

  edm::Handle<ESDigiCollection> EcalDigiES;

  e.getByToken( ESdigiCollectionToken_ , EcalDigiES );

  // Return if no preshower data
  if( !EcalDigiES.isValid() ) return;

  // PRESHOWER
  
  // loop over Digis

  const ESDigiCollection * preshowerDigi = EcalDigiES.product () ;

  std::vector<double> esADCCounts ;
  esADCCounts.reserve(ESDataFrame::MAXSAMPLES);

  int nDigis = 0;

  for (unsigned int digis=0; digis<EcalDigiES->size(); ++digis) {

    ESDataFrame esdf=(*preshowerDigi)[digis];
    int nrSamples=esdf.size();
    
    ESDetId esid = esdf.id () ;
    
    nDigis++;
    
    for (int sample = 0 ; sample < nrSamples; ++sample) {
      esADCCounts[sample] = 0.;
    }
    
    for (int sample = 0 ; sample < nrSamples; ++sample) {
      ESSample mySample = esdf[sample];
      esADCCounts[sample] = (mySample.adc()) ;
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
  
  if ( meESDigiMultiplicity_ ) meESDigiMultiplicity_->Fill(nDigis);
  
}

                                                                                                                                                             
