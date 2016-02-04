/*
 * \file EcalPreshowerNoiseDistrib.cc
 *
*/

#include <Validation/EcalDigis/interface/EcalPreshowerNoiseDistrib.h>
#include "DQMServices/Core/interface/DQMStore.h"
using namespace cms;
using namespace edm;
using namespace std;

EcalPreshowerNoiseDistrib::EcalPreshowerNoiseDistrib(const ParameterSet& ps):
  ESdigiCollection_(ps.getParameter<edm::InputTag>("ESdigiCollection"))
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
  
  // histos
  meESDigiMultiplicity_=0;
  for (int ii=0; ii<3; ii++ ) { meESDigiADC_[ii] = 0; }

  Char_t histo[200];
  if ( dbe_ ) {
    sprintf (histo, "multiplicity" ) ;
    meESDigiMultiplicity_ = dbe_->book1D(histo, histo, 1000, 0., 137728);
    
    for ( int ii = 0; ii < 3 ; ii++ ) {
      sprintf (histo, "esRefHistos%02d", ii) ;
      meESDigiADC_[ii] = dbe_->book1D(histo, histo, 35, 983.5, 1018.5) ;
    }
    
    for ( int ii = 0; ii < 3 ; ii++ ) {
      sprintf (histo, "esRefHistosCorr%02d", ii) ;
      meESDigiCorr_[ii] = dbe_->book2D(histo, histo, 35, 983.5, 1018.5, 35, 983.5, 1018.5) ;
    }
        
    meESDigi3D_ = dbe_->book3D("meESDigi3D_", "meESDigi3D_", 35, 983.5, 1018.5, 35, 983.5, 1018.5, 35, 983.5, 1018.5) ;
  }
}


void EcalPreshowerNoiseDistrib::analyze(const Event& e, const EventSetup& c){

  Handle<ESDigiCollection> EcalDigiES;
  
  e.getByLabel( ESdigiCollection_ , EcalDigiES );

  // retrun if no data
  if( !EcalDigiES.isValid() ) return;
  
  // loop over Digis
  const ESDigiCollection * preshowerDigi = EcalDigiES.product () ;
  
  std::vector<double> esADCCounts ;
  esADCCounts.reserve(ESDataFrame::MAXSAMPLES);
  
  int nDigis = 0;

  for (unsigned int digis=0; digis<EcalDigiES->size(); ++digis) {
    nDigis++;
    ESDataFrame esdf=(*preshowerDigi)[digis];
    int nrSamples=esdf.size();
    for (int sample = 0 ; sample < nrSamples; ++sample) {
      ESSample mySample = esdf[sample];
      if (meESDigiADC_[sample]) { meESDigiADC_[sample] ->Fill(mySample.adc()); }
    }

    // to study correlations
    if(meESDigiCorr_[0]){ meESDigiCorr_[0]->Fill(esdf[0].adc(),esdf[1].adc()); } 
    if(meESDigiCorr_[1]){ meESDigiCorr_[1]->Fill(esdf[0].adc(),esdf[2].adc()); } 
    if(meESDigiCorr_[2]){ meESDigiCorr_[2]->Fill(esdf[1].adc(),esdf[2].adc()); } 

    // reference histo: sample0, sample1, sample2
    if ( meESDigi3D_ ) meESDigi3D_ -> Fill(esdf[0].adc(),esdf[1].adc(),esdf[2].adc());
  }
  
  if ( meESDigiMultiplicity_ ) meESDigiMultiplicity_->Fill(nDigis);
  
}

