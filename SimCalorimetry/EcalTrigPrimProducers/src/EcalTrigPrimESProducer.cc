// user include files
#include "SimCalorimetry/EcalTrigPrimProducers/interface/EcalTrigPrimESProducer.h"

#include <iostream>
#include <fstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// //
// // constructors and destructor
// //
// EcalTrigPrimESProducer::EcalTrigPrimESProducer(const edm::ParameterSet& iConfig) :
//   dbFilenameEB_(iConfig.getUntrackedParameter<std::string>("DatabaseFileEB","")),dbFilenameEE_(iConfig.getUntrackedParameter<std::string>("DatabaseFileEE",""))
// {
//   //the following line is needed to tell the framework what
//   // data is being produced
//   setWhatProduced(this);
  
//   //now do what ever other initialization is needed
// }


// EcalTrigPrimESProducer::~EcalTrigPrimESProducer()
// { 
// }

// //
// // member functions
// //

// // ------------ method called to produce the data  ------------
// EcalTrigPrimESProducer::ReturnType

// EcalTrigPrimESProducer::produce(const EcalTPParametersRcd& iRecord)
// {

//   using namespace edm::es;
//   std::auto_ptr<EcalTPParameters> prod(new EcalTPParameters());

//   parseTextFile(*prod);

//   return prod;
// }

// void EcalTrigPrimESProducer::parseTextFile(EcalTPParameters& ecaltpp)
// {

//   //FIXME ...or wait for DB
//   // for the moment we have only barrel parameters
//   int NBphysparams  = 4 ;
//   int NBxtalparams  = 9 ;
//   int NBstripparams = 6 ;
//   int NBtowerparams = 1029 ;

//   std::vector<unsigned int> param ;
//   std::vector<float> paramF ;

//   std::string filenameEB="SimCalorimetry/EcalTrigPrimProducers/data/"+dbFilenameEB_;
//   edm::FileInPath fileInPath(filenameEB);
//   std::ifstream infile (fileInPath.fullPath().c_str()) ;
//   if (infile.is_open()) {


//     // phys structure : xtalLSB (GeV), EtSaturation (GeV), ttf_threshold_Low (GeV), ttf_threshold_High (GeV)
//     // xtal structure : ped, mult, shift [gain12] , ped, mult ,shift [gain6], ... [gain1]
//     // strip structure : sliding , weight1, weight2, weight3, weight4, weight5
//     // tower structure : lut[0], ... , lut[1023], el, eh, tl, th, lut_fg

//     unsigned int data ;
//     float dataF;
//     std::string dataCard ;
//     int smNb, towerNbInSm, stripNbInTower, xtalNbInStrip ;

//     while (!infile.eof()) {
//       infile>>dataCard ;
 
//       if (dataCard == "PHYSICS") {
// 	paramF.clear() ;
// 	for (int i=0 ; i <NBphysparams ; i++) {
// 	  //	  infile>>std::dec>>data ;
// 	  infile>>dataF ;
// 	  paramF.push_back(dataF) ;
// 	}
// 	ecaltpp.setPhysicsParameters(paramF);
//       }


//       if (dataCard == "CRYSTAL") {
// 	infile>>std::dec>>smNb>>towerNbInSm>>stripNbInTower>>xtalNbInStrip ;
// 	param.clear() ;
// 	for (int i=0 ; i <NBxtalparams ; i++) {
// 	  infile>>std::hex>>data ;
// 	  param.push_back(data) ;
// 	}
// 	std::vector<int> range = getRange(smNb, towerNbInSm, stripNbInTower, xtalNbInStrip) ;
// 	for (int SM = range[0] ; SM < range[1] ; SM++)
// 	  for (int towerInSM = range[2] ; towerInSM < range[3] ; towerInSM++)
// 	    for (int stripInTower = range[4] ; stripInTower < range[5] ;  stripInTower++)
// 	      for (int xtalInStrip =  range[6] ; xtalInStrip <  range[7] ; xtalInStrip++) {
// 		//		int index = ((68*SM + towerInSM)*5 + stripInTower)*5 +  xtalInStrip ;		
// 		//		xtalParam_[index] = param ;
// 		//		ecaltpp.setXtalParameters(index,param);
// 		ecaltpp.setXtalParameters(SM,towerInSM,stripInTower,xtalInStrip,param);
// 	      }
//       }

//       if (dataCard == "STRIP") {
// 	infile>>std::dec>>smNb>>towerNbInSm>>stripNbInTower ;
// 	param.clear() ;
// 	for (int i=0 ; i <NBstripparams ; i++) {
// 	  infile>>std::hex>>data ;
// 	  param.push_back(data) ;
// 	}
// 	std::vector<int> range = getRange(smNb, towerNbInSm, stripNbInTower) ;
// 	for (int SM = range[0] ; SM < range[1] ; SM++)
// 	  for (int towerInSM = range[2] ; towerInSM < range[3] ; towerInSM++)
// 	    for (int stripInTower = range[4] ; stripInTower < range[5] ;  stripInTower++) {
// 	      //	      int index = (68*SM + towerInSM)*5 + stripInTower ;
// 	      //	      stripParam_[index] = param ;
// 	      //		ecaltpp.setStripParameters(index,param);
// 		ecaltpp.setStripParameters(SM,towerInSM,stripInTower,param);
// 	    }
//       }

//       if (dataCard == "TOWER") {
// 	infile>>std::dec>>smNb>>towerNbInSm ;
// 	param.clear() ;
// 	for (int i=0 ; i <NBtowerparams ; i++) {
// 	  infile>>std::hex>>data ;
// 	  param.push_back(data) ;
// 	}	
// 	std::vector<int> range = getRange(smNb, towerNbInSm) ;
// 	for (int SM = range[0] ; SM < range[1] ; SM++)
// 	  for (int towerInSM = range[2] ; towerInSM < range[3] ; towerInSM++) {
// 	    //int index = 68*SM + towerInSM ;	          
// 	    //	    towerParam_[index] = param ;
// 	    //		ecaltpp.setTowerParameters(index,param);
// 		ecaltpp.setTowerParameters(SM,towerInSM,param);
// 	  }
//       }

//     }
//   }
//   infile.close();
// }

// std::vector<int> EcalTrigPrimESProducer::getRange(int smNb, int towerNbInSm, int stripNbInTower, int xtalNbInStrip)
// {
//   std::vector<int> range ;
//   range.push_back(1)  ; // smNbMin
//   range.push_back(37) ; // smNbMax
//   range.push_back(1)  ; // towerNbMin
//   range.push_back(69) ; // towerNbMax
//   range.push_back(1)  ; // stripNbMin
//   range.push_back(6)  ; // stripNbMax
//   range.push_back(1)  ; // xtalNbMin
//   range.push_back(6)  ; // xtalNbMax

//   if (smNb>0) {
//     range[0] = smNb ; 
//     range[1] = smNb+1 ;
//   }
//   if (towerNbInSm>0) {
//     range[2] = towerNbInSm ; 
//     range[3] = towerNbInSm+1 ;
//   }
//   if (stripNbInTower>0) {
//     range[4] = stripNbInTower ; 
//     range[5] = stripNbInTower+1 ;
//   }
//   if (xtalNbInStrip>0) {
//     range[6] = xtalNbInStrip ; 
//     range[7] = xtalNbInStrip+1 ;
//   }

//   return range ;
// }
