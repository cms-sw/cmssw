/** \class EcalEBTrigPrimBaseAlgo
 *
 * EcalEBTrigPrimBaseAlgo 
 * starting point for Phase II: build TPs out of Phase I digis to start building the
 * infrastructures
 *
 *
 ************************************************************/
#include <string>
#include <algorithm>
#include <numeric>
#include <functional>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBTrigPrimBaseAlgo.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"

#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"
#include "CondFormats/DataRecord/interface/EcalTPGPedestalsRcd.h"
//----------------------------------------------------------------------

const unsigned int EcalEBTrigPrimBaseAlgo::nrSamples_=5;
const unsigned int EcalEBTrigPrimBaseAlgo::maxNrTowers_=2448;
const unsigned int EcalEBTrigPrimBaseAlgo::maxNrSamplesOut_=10;




//----------------------------------------------------------------------
void EcalEBTrigPrimBaseAlgo::init(const edm::EventSetup & setup) {
  if (!barrelOnly_) {
    setup.get<CaloGeometryRecord>().get( theGeometry_ );
    setup.get<IdealGeometryRecord>().get(eTTmap_);
  }

  // initialise data structures
  initStructures(towerMapEB_);
  hitTowers_.resize(maxNrTowers_);


  linearizer_.resize(nbMaxXtals_);
  for (int i=0;i<nbMaxXtals_;i++) linearizer_[i] = new  EcalFenixLinearizer(famos_);

  //
  std::vector <int> v;
  v.resize(maxNrSamples_);
  lin_out_.resize(nbMaxXtals_);  
  for (int i=0;i<nbMaxXtals_;i++) lin_out_[i]=v;
  //
  amplitude_filter_ = new EcalFenixAmplitudeFilter();
  filt_out_.resize(maxNrSamples_);
  peak_out_.resize(maxNrSamples_);
  // these two are dummy
  fgvb_out_.resize(maxNrSamples_);
  fgvb_out_temp_.resize(maxNrSamples_);  
  //
  peak_finder_ = new  EcalFenixPeakFinder();
  fenixFormatterEB_ = new EcalFenixStripFormatEB();
  format_out_.resize(maxNrSamples_);
 
}
//----------------------------------------------------------------------

EcalEBTrigPrimBaseAlgo::~EcalEBTrigPrimBaseAlgo() 
{ }



  



//----------------------------------------------------------------------

int  EcalEBTrigPrimBaseAlgo::findStripNr(const EBDetId &id){
  int stripnr;
  int n=((id.ic()-1)%100)/20; //20 corresponds to 4 * ecal_barrel_crystals_per_strip FIXME!!
  if (id.ieta()<0) stripnr = n+1;
  else stripnr =nbMaxStrips_ - n; 
  return stripnr;
}

