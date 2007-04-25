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
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalTrigPrimFunctionalAlgo.h"
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixLinearizer.h"
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStrip.h"
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcp.h"

//#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"

#include "CondFormats/DataRecord/interface/EcalTPParametersRcd.h"

#include <TTree.h>
#include <TMath.h>
//----------------------------------------------------------------------

EcalTrigPrimFunctionalAlgo::EcalTrigPrimFunctionalAlgo(const edm::EventSetup & setup,int binofmax,int nrsamples, bool tcpFormat, bool barrelOnly,bool debug, double ebDccAdcToGeV,double eeDccAdcToGeV):
  valid_(false),valTree_(NULL),binOfMaximum_(binofmax),nrSamplesToWrite_(nrsamples),
  tcpFormat_(tcpFormat), barrelOnly_(barrelOnly), debug_(debug),
  ebDccAdcToGeV_(ebDccAdcToGeV),eeDccAdcToGeV_(eeDccAdcToGeV)

{this->init(setup);}

//----------------------------------------------------------------------
EcalTrigPrimFunctionalAlgo::EcalTrigPrimFunctionalAlgo(const edm::EventSetup & setup,TTree *tree,int binofmax, int nrsamples,bool tcpFormat, bool barrelOnly, bool debug, double ebDccAdcToGeV,double eeDccAdcToGeV):
  valid_(true),valTree_(tree),binOfMaximum_(binofmax),nrSamplesToWrite_(nrsamples),
  tcpFormat_(tcpFormat), barrelOnly_(barrelOnly),debug_(debug),
  ebDccAdcToGeV_(ebDccAdcToGeV),eeDccAdcToGeV_(eeDccAdcToGeV)

{this->init(setup);}

//----------------------------------------------------------------------
void EcalTrigPrimFunctionalAlgo::init(const edm::EventSetup & setup) {
  //FIXME: check validities
  if (!barrelOnly_) {
    edm::ESHandle<CaloGeometry> theGeometry;
    edm::ESHandle<CaloSubdetectorGeometry> theEndcapGeometry_handle;
    setup.get<IdealGeometryRecord>().get( theGeometry );
    setup.get<IdealGeometryRecord>().get("EcalEndcap",theEndcapGeometry_handle);
    theEndcapGeometry = &(*theEndcapGeometry_handle);
    setup.get<IdealGeometryRecord>().get(eTTmap_);
  }
  edm::ESHandle<EcalTPParameters> theEcalTPParameters_handle;
  setup.get<EcalTPParametersRcd>().get(theEcalTPParameters_handle);
  ecaltpp_=theEcalTPParameters_handle.product();


  // endcap mapping
  edm::ESHandle< EcalElectronicsMapping > ecalmapping;
  setup.get< EcalMappingRcd >().get(ecalmapping);
  theMapping_ = ecalmapping.product();

  //create main sub algos
  estrip_= new EcalFenixStrip(valTree_,ecaltpp_,theMapping_,debug_);
  etcp_ = new EcalFenixTcp(ecaltpp_,tcpFormat_,debug_) ;
}
//----------------------------------------------------------------------

EcalTrigPrimFunctionalAlgo::~EcalTrigPrimFunctionalAlgo() 
{
    delete estrip_;
    delete etcp_;
}
//----------------------------------------------------------------------
void EcalTrigPrimFunctionalAlgo::updateESRecord(double ttfLowEB, double ttfHighEB, double ttfLowEE, double ttfHighEE)
{
  const_cast <EcalTPParameters *> (ecaltpp_)->changeThresholds(ttfLowEB, ttfHighEB, ttfLowEE, ttfHighEE);
}
//----------------------------------------------------------------------

// void EcalTrigPrimFunctionalAlgo::fillMap(const EBDigiCollection * col,
//                                          std::map<EcalTrigTowerDetId,std::vector<std::vector<const EBDataFrame * > >,std::less<EcalTrigTowerDetId> >  & towerMap,
// 					 int &nhits) 
//   // implementation for Barrel
// {
//   typedef std::map<EcalTrigTowerDetId,std::vector<std::vector<const EBDataFrame * > >,std::less<EcalTrigTowerDetId> > TOWMAP;

//   // loop over dataframes and fill map 
//   if (col) {
//     for(unsigned int i = 0; i < col->size() ; ++i) {
//       const EBDataFrame &samples = (*col)[i];
//       const EBDetId & myId=samples.id();
//       const EcalTrigTowerDetId coarser= myId.tower();
// /*    if(coarser.null())   */
// /*      { */
// /* 	  LogDebug("")<< "Cell " << samples << " has trivial coarser granularity (probably EFRY corner, not in this tower map; hit ignored)"; */
// /* 	  continue; */
// /* 	}	 */
	
//  // here we store a vector of EBDataFrames for each strip into a vector belonging to the corresponding tower
 
//       nhits++;
//       int n=(((samples.id()).ic()-1)%100)/20; //20 corresponds to 4 * ecal_barrel_crystals_per_strip FIXME!!
//       int stripnr;
//       if ((samples.id()).ieta()<0) stripnr = n+1;
//       else stripnr =ecal_barrel_strips_per_trigger_tower - n; //FIXME: take from official place

//       TOWMAP::iterator it= towerMap.find(coarser);
//       if(it==towerMap.end()) {
//         for (int i=0;i<ecal_barrel_strips_per_trigger_tower;i++ ) {
//           std::vector<const EBDataFrame *>  truc;
//           towerMap[coarser].push_back(truc);
//         } 
//       }
//       const EBDataFrame * p=& samples;
//       (towerMap[coarser])[stripnr-1].push_back(p);
//     }
//   }
//   LogDebug("EcalTPG")<< "[EcalTrigPrimFunctionalAlgo] found " << nhits << " frames in " 
//   		<< towerMap.size() << " Barrel towers  ";

// }
//---------------------------------------------------------------------------------------------------------------   
// void EcalTrigPrimFunctionalAlgo::fillMap(const EEDigiCollection * col,
//                                          std::map<EcalTrigTowerDetId,std::vector<std::vector<const EEDataFrame * > >,std::less<EcalTrigTowerDetId> >  & towerMap,
// 					 int &nhits) 
// {
//   // implementation for endcap
//   // Muriel's temporary version, waiting for geometry of pseudostrips
//   // for the moment we put into this map for each TT  :
//   // the first pseudo strip created with the first five crystals, 
//   // the 2nd pseudostrip created with the  next five crystals and so on ...

//   typedef std::map<EcalTrigTowerDetId,std::vector<std::vector<const EEDataFrame * > >,std::less<EcalTrigTowerDetId> > TOWMAP;
  
//   int nbOfPseudoStrips=0;
//   int nbOfCrystals=0;
  
//   if (col) {
//     LogDebug("EcalTPG") <<"Fill endcap mapping, EECollection size = "<<col->size();
//     for(unsigned int i = 0; i < col->size() ; ++i) {
//       const EEDataFrame &samples = (*col)[i];
//       const EEDetId & myId=samples.id();
//       EcalTrigTowerDetId coarser=(*eTTmap_).towerOf(myId);
//       const EcalTriggerElectronicsId elId = theMapping_->getTriggerElectronicsId(myId);
//       //      printf(" tcc, towerId, pseudostrip , channelId = %d, %d %d %d\n",elId.tccId(),elId.ttId(),elId.pseudoStripId(),elId.channelId());fflush(stdout);
//       int stripnr=elId.pseudoStripId();
//       nhits++;
//       TOWMAP::iterator it= towerMap.find(coarser);
//       if(it==towerMap.end()) {
//         for (int i=0;i<ecal_endcap_max_strips_per_trigger_tower;i++ ) {
//           std::vector<const EEDataFrame *>  truc;
//           towerMap[coarser].push_back(truc);
// 	  //	  printf (" created %d entries for this tower\n",ecal_endcap_max_strips_per_trigger_tower);
//         } 
//       }
//       const EEDataFrame * p=& samples;
//       if ((towerMap[coarser])[stripnr-1].size()<5 ) {
//       (towerMap[coarser])[stripnr-1].push_back(p);
//       }else {
//       //      std::cout <<"tower "<<coarser<<" Detid "<<myId<<", strip "<<stripnr<<" size "<<(towerMap[coarser])[stripnr-1].size()<<std::endl;
// 	std::cout <<" !!!!!!!!!!!!! Too many xtals for TT "<<coarser<<" stripnr "<<stripnr<<std::endl;
// 	for (int kk=0;kk<(towerMap[coarser])[stripnr-1].size();kk++)
// 	  std::cout<<"xtal "<<kk<<" detid "<<((towerMap[coarser])[stripnr-1])[kk]->id()<<std::endl;
//       }
    
//       //ENDCAP:missing geometry  -  no EEDetId::tower()
//       //       EcalTrigTowerDetId coarser=(*eTTmap_).towerOf(myId);
//       // /*    if(coarser.null())   */
//       // /*      { */
//       // /* 	  LogDebug("")<< "Cell " << samples << " has trivial coarser granularity (probably EFRY corner, not in this tower map; hit ignored)"; */
//       // /* 	  continue; */
//       // /* 	}           */

//       //       nhits++;
//       //       TOWMAP::iterator it = towerMap.find(coarser);
//       //       std::vector<const EEDataFrame *> crystalData;
  
//       //       if(it==towerMap.end())   //triggerTower not yet found - Icreate the TT element in the map
//       //         towerMap[coarser].push_back(crystalData);
    
//       //         nbOfPseudoStrips = towerMap[coarser].size();
//       //         nbOfCrystals = (towerMap[coarser])[nbOfPseudoStrips-1].size();
  
//       //       if (nbOfCrystals == ecal_endcap_maxcrystals_per_strip) {
//       //         towerMap[coarser].push_back(crystalData);
//       //         nbOfPseudoStrips = towerMap[coarser].size();
//       //       }  
//       //       const EEDataFrame *p=& samples;
//       //       (towerMap[coarser])[nbOfPseudoStrips-1].push_back(p);
//   }
//     LogDebug("EcalTPG")<<"fillMap-Endcap"<<"[EcalTrigPrimFunctionalAlgo] (found " 
// 		       << nhits << " frames in "<< towerMap.size() << " Endcap towers ";
//   }
//   else {
//     LogDebug("EcalTPG")<<"FillEndcap - Fill endcap EECollection size=0";
//   }
      
// }
//-----------------------------------------------------------------------

// //----------------------------------------------------------------------
// int EcalTrigPrimFunctionalAlgo::calculateTTF(const int en) {
//   //temporary version of TTF calculation for Endcap
//   //  int high=83; // adc value corresponding to 5 GeV, factor 0.06
//   //  int low=42;  // adc value corresponding to 2.5 GeV, factor 0.06
//   // temporary for temporary Endcap version !!!

//   double threshLow_ =2.5; //GeV
//   double threshHigh_=5.;
//   //  
//   int high=int(threshHigh_/0.06);
//   int low=int(threshLow_/0.06);
//   int ttf=0;
//   if (en>high) ttf=3;
//   else if (ttf<high && ttf >low ) ttf=2;
//   return ttf;
// }
//----------------------------------------------------------------------
int EcalTrigPrimFunctionalAlgo::findTowerNrInTcc(const EcalTrigTowerDetId &id)
{
  if (id.subDet()== EcalBarrel) { // finds tower nr in TCC   //FIXME: still correct?
   const int nrphis=4;
   int ieta=id.ietaAbs();
    int iphi=id.iphi();
    int basenr=(ieta-1)*nrphis +1;
    int towernr=basenr+(iphi-1)%nrphis;
    return  towernr;
  } 
  else if (id.subDet()==EcalEndcap) {
    return theMapping_->iTT(id);
  }
  else {
    LogDebug("EcalTPG")<<"Wrong EcalTrigTowerDetId ";
    return 0;
  }
}
//----------------------------------------------------------------------
int EcalTrigPrimFunctionalAlgo::findTccNr(const EcalTrigTowerDetId &id)
{
// finds Tcc Nr
  if (id.subDet()== EcalBarrel) { 
    return EcalTPParameters::nrMinTccEB_; //FIXME
  }
  else if (id.subDet()==EcalEndcap) {
    return theMapping_->TCCid(id);
  }
  else {
    LogDebug("EcalTPG")<<"Wrong EcalTrigTowerDetId ";
    return 0;
  }     
} 
//----------------------------------------------------------------------
int  EcalTrigPrimFunctionalAlgo::findStripNr(const EBDetId &id){
      int stripnr;
      int n=((id.ic()-1)%100)/20; //20 corresponds to 4 * ecal_barrel_crystals_per_strip FIXME!!
      if (id.ieta()<0) stripnr = n+1;
      //      else stripnr =ecal_barrel_strips_per_trigger_tower - n; 
      else stripnr =EcalTPParameters::nbMaxStrips_ - n; 
      return stripnr;
}
//----------------------------------------------------------------------
int  EcalTrigPrimFunctionalAlgo::findStripNr(const EEDetId &id){
      int stripnr;
      const EcalTriggerElectronicsId elId = theMapping_->getTriggerElectronicsId(id);
      stripnr=elId.pseudoStripId();
      return stripnr;
}
