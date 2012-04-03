/** \class EcalTrigPrimFunctionalAlgo
 *
 * EcalTrigPrimFunctionalAlgo is the main algorithm class for TPG
 * It coordinates all the aother algorithms
 * Structure is very close to electronics
 *
 *
 * \author Ursula Berthon, Stephanie Baffioni,  LLR Palaiseau
 *
 * \version   1st Version may 2006
 * \version   2nd Version jul 2006

 *
 ************************************************************/
#include <string>
#include <algorithm>
#include <numeric>
#include <functional>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalTrigPrimFunctionalAlgo.h"
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixLinearizer.h"
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStrip.h"
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcp.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"

#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"
#include "CondFormats/DataRecord/interface/EcalTPGPedestalsRcd.h"

#include <TTree.h>
#include <TMath.h>

const unsigned int EcalTrigPrimFunctionalAlgo::nrSamples_=5;  //to be written
const unsigned int EcalTrigPrimFunctionalAlgo::maxNrSamplesOut_=10;  
const unsigned int EcalTrigPrimFunctionalAlgo::maxNrTowers_=2448;
const unsigned int EcalTrigPrimFunctionalAlgo::maxNrTPs_=2448; //FIXME??

//----------------------------------------------------------------------

EcalTrigPrimFunctionalAlgo::EcalTrigPrimFunctionalAlgo(const edm::EventSetup & setup,int binofmax,bool tcpFormat, bool barrelOnly,bool debug,bool famos): binOfMaximum_(binofmax),
  tcpFormat_(tcpFormat), barrelOnly_(barrelOnly), debug_(debug), famos_(famos)

{
 if (famos_) maxNrSamples_=1;  //get from input??
 else maxNrSamples_=10;
 this->init(setup);
}

//----------------------------------------------------------------------
void EcalTrigPrimFunctionalAlgo::init(const edm::EventSetup & setup) {
  if (!barrelOnly_) {
    edm::ESHandle<CaloGeometry> theGeometry;
    edm::ESHandle<CaloSubdetectorGeometry> theEndcapGeometry_handle;
    setup.get<CaloGeometryRecord>().get( theGeometry );
    setup.get<EcalEndcapGeometryRecord>().get("EcalEndcap",theEndcapGeometry_handle);
    theEndcapGeometry = &(*theEndcapGeometry_handle);
    setup.get<IdealGeometryRecord>().get(eTTmap_);
  }
  // endcap mapping
  edm::ESHandle< EcalElectronicsMapping > ecalmapping;
  setup.get< EcalMappingRcd >().get(ecalmapping);
  theMapping_ = ecalmapping.product();

  //create main sub algos
  estrip_= new EcalFenixStrip(setup,theMapping_,debug_,famos_,maxNrSamples_,nbMaxXtals_);
  etcp_ = new EcalFenixTcp(setup,tcpFormat_,debug_,famos_,binOfMaximum_,maxNrSamples_,nbMaxStrips_) ;

  // initialise data structures
  initStructures(towerMapEB_);
  initStructures(towerMapEE_);

  hitTowers_.resize(maxNrTowers_);
  towtp_.resize(maxNrSamplesOut_);
  towtp2_.resize(maxNrSamplesOut_);
  
}
//----------------------------------------------------------------------

EcalTrigPrimFunctionalAlgo::~EcalTrigPrimFunctionalAlgo() 
{
    delete estrip_;
    delete etcp_;
}
//----------------------------------------------------------------------
void EcalTrigPrimFunctionalAlgo::run(const edm::EventSetup & setup, EBDigiCollection const * col,
				     EcalTrigPrimDigiCollection & result,
				     EcalTrigPrimDigiCollection & resultTcp)
{
  run_part1_EB(col);
  run_part2(setup,col,towerMapEB_,result,resultTcp);
}

//----------------------------------------------------------------------
void EcalTrigPrimFunctionalAlgo::run(const edm::EventSetup & setup, EEDigiCollection const * col,
                                                        EcalTrigPrimDigiCollection & result,
							EcalTrigPrimDigiCollection & resultTcp)
{

  run_part1_EE(col);
  run_part2(setup, col,towerMapEE_,result,resultTcp);
}
//----------------------------------------------------------------------
int  EcalTrigPrimFunctionalAlgo::findStripNr(const EBDetId &id){
  int stripnr;
  int n=((id.ic()-1)%100)/20; //20 corresponds to 4 * ecal_barrel_crystals_per_strip FIXME!!
  if (id.ieta()<0) stripnr = n+1;
  else stripnr =nbMaxStrips_ - n; 
  return stripnr;
}
//----------------------------------------------------------------------
int  EcalTrigPrimFunctionalAlgo::findStripNr(const EEDetId &id){
  int stripnr;
  const EcalTriggerElectronicsId elId = theMapping_->getTriggerElectronicsId(id);
  stripnr=elId.pseudoStripId();
  return stripnr;
}
//----------------------------------------------------------------------

void EcalTrigPrimFunctionalAlgo::run_part1_EB(EBDigiCollection const * col){
  clean(towerMapEB_);
  // loop over dataframes and fill map 
  fillMap(col,towerMapEB_);
}
//----------------------------------------------------------------------
void EcalTrigPrimFunctionalAlgo::run_part1_EE(EEDigiCollection const * col){
  clean(towerMapEE_);
  // loop over dataframes and fill map 
  fillMap(col,towerMapEE_);
}
