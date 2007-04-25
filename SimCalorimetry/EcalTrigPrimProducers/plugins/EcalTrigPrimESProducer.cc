// user include files
#include "EcalTrigPrimESProducer.h"

#include <iostream>
#include <fstream>
#include <TMath.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
  const int EcalTrigPrimESProducer::MIN_TCC_EB =37;
  const int EcalTrigPrimESProducer::MAX_TCC_EB =73;
  const int EcalTrigPrimESProducer::MIN_TCC_EE_PLUS =73;
  const int EcalTrigPrimESProducer::MAX_TCC_EE_PLUS =109;
  const int EcalTrigPrimESProducer::MIN_TCC_EE_MINUS =1;
  const int EcalTrigPrimESProducer::MAX_TCC_EE_MINUS =37;
  const int EcalTrigPrimESProducer::MIN_TT_EB = 1;
  const int EcalTrigPrimESProducer::MAX_TT_EB = 69;
  const int EcalTrigPrimESProducer::MIN_TT_EE = 1; 
//FIXME: const int EcalTrigPrimESProducer::MAX_TT_EE = 25; //This is a maximum from outer (=16) and inner (=24 without 4 virtual ones)
  const int EcalTrigPrimESProducer::MAX_TT_EE = 29; //temporary
  const int EcalTrigPrimESProducer::MIN_STRIP_EB = 1;
  const int EcalTrigPrimESProducer::MAX_STRIP_EB = 6;
  const int EcalTrigPrimESProducer::MIN_STRIP_EE = 1;
  const int EcalTrigPrimESProducer::MAX_STRIP_EE = 6;
  const int EcalTrigPrimESProducer::MIN_XTAL_EB = 1;
  const int EcalTrigPrimESProducer::MAX_XTAL_EB = 6;
  const int EcalTrigPrimESProducer::MIN_XTAL_EE = 1;
  const int EcalTrigPrimESProducer::MAX_XTAL_EE = 6;

EcalTrigPrimESProducer::EcalTrigPrimESProducer(const edm::ParameterSet& iConfig) :
  dbFilenameEB_(iConfig.getUntrackedParameter<std::string>("DatabaseFileEB","")),dbFilenameEE_(iConfig.getUntrackedParameter<std::string>("DatabaseFileEE",""))
{
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  
  //now do what ever other initialization is needed
}


EcalTrigPrimESProducer::~EcalTrigPrimESProducer()
{ 
}

//
// member functions
//

// ------------ method called to produce the data  ------------
EcalTrigPrimESProducer::ReturnType

EcalTrigPrimESProducer::produce(const EcalTPParametersRcd& iRecord)
{

  using namespace edm::es;
  std::auto_ptr<EcalTPParameters> prod(new EcalTPParameters());
  prod->setConstants(TMath::Max(MAX_TT_EB,MAX_TT_EE)-1,
		     TMath::Max(MAX_STRIP_EB,MAX_STRIP_EE)-1,
		     TMath::Max(MAX_XTAL_EB,MAX_XTAL_EE)-1,
		     MIN_TCC_EB,MAX_TCC_EB-1);
  parseTextFile(*prod);
  return prod;
}

void EcalTrigPrimESProducer::parseTextFile(EcalTPParameters& ecaltpp)
{

  // phys structure : xtalLSB (GeV), EtSaturation (GeV), ttf_threshold_Low (GeV), ttf_threshold_High (GeV)
  // EB :
  // xtal structure : ped, mult, shift [gain12] , ped, mult ,shift [gain6], ... [gain1]
  // strip structure : sliding , weight1, weight2, weight3, weight4, weight5
  // tower structure : lut[0], ... , lut[1023], el, eh, tl, th, lut_fg


  // EE :
  // xtal structure : ped, mult, shift [gain12] , ped, mult ,shift [gain6], ... [gain1]
  // strip structure : sliding , weight1, weight2, weight3, weight4, weight5, threshold_fg, strip_lut_fg
  // tower structure : lut[0], ... , lut[1023], tower_lut_fg

  // subdet=0 => Barrel, subdet=1 => EndCap

  //FIXME ...or wait for DB

  int NBphysparams  = 4 ;
  int NBxtalparams  = 9 ;
  int NBstripparams[2] = {6, 8} ;
  int NBtowerparams[2] = {1029, 1025} ;

  std::ifstream infile[2] ;

  std::vector<unsigned int> param ;
  std::vector<float> paramF ;

  unsigned int data ;
  float dataF;
  std::string dataCard ;
  int tccNb, towerNbInTcc, stripNbInTower, xtalNbInStrip ;
  std::string filename ;

  for (int subdet=0 ; subdet<2 ; subdet++) {
    filename="SimCalorimetry/EcalTrigPrimProducers/data/"+dbFilenameEB_;
    if (subdet == 1) filename="SimCalorimetry/EcalTrigPrimProducers/data/"+dbFilenameEE_ ;
    edm::FileInPath fileInPath(filename);
    infile[subdet].open(fileInPath.fullPath().c_str()) ;

    if (infile[subdet].is_open()) {
      while (!infile[subdet].eof()) {
	infile[subdet]>>dataCard ;

 
	if (dataCard == "PHYSICS") {
	  paramF.clear() ;
	  for (int i=0 ; i <NBphysparams ; i++) {
	    //	  infile>>std::dec>>data ;
	    infile[subdet]>>dataF ;
	    paramF.push_back(dataF) ;
	  }
	  ecaltpp.setPhysicsParameters(paramF);
	}


	if (dataCard == "CRYSTAL") {
	  infile[subdet]>>std::dec>>tccNb>>towerNbInTcc>>stripNbInTower>>xtalNbInStrip ;
	  param.clear() ;
	  for (int i=0 ; i <NBxtalparams ; i++) {
	    infile[subdet]>>std::hex>>data ;
	    param.push_back(data) ;
	  }
	  std::vector<int> range = getRange(subdet, tccNb, towerNbInTcc, stripNbInTower, xtalNbInStrip) ;
	  for (int TCC = range[0] ; TCC < range[1] ; TCC++)
	    for (int towerInTCC = range[2] ; towerInTCC < range[3] ; towerInTCC++)
	      for (int stripInTower = range[4] ; stripInTower < range[5] ;  stripInTower++)
		for (int xtalInStrip =  range[6] ; xtalInStrip <  range[7] ; xtalInStrip++) {
		  //		int index = ((68*TCC + towerInTCC)*5 + stripInTower)*5 +  xtalInStrip ;		
		  //		xtalParam_[index] = param ;
		  //		ecaltpp.setXtalParameters(index,param);
		  // the same index can be used for both EB and EE since there are always less than 68 trigger towers
		  ecaltpp.setXtalParameters(TCC,towerInTCC,stripInTower,xtalInStrip,param);
		}
      
	  if (subdet!=0) { // repeat for EE eta<0
	    std::vector<int> range = getRange(-subdet, tccNb, towerNbInTcc, stripNbInTower, xtalNbInStrip) ;
	    for (int TCC = range[0] ; TCC < range[1] ; TCC++)
	      for (int towerInTCC = range[2] ; towerInTCC < range[3] ; towerInTCC++)
		for (int stripInTower = range[4] ; stripInTower < range[5] ;  stripInTower++)
		  for (int xtalInStrip =  range[6] ; xtalInStrip <  range[7] ; xtalInStrip++) {
		    //		int index = ((68*TCC + towerInTCC)*5 + stripInTower)*5 +  xtalInStrip ;		
		    //		xtalParam_[index] = param ;
		    //		ecaltpp.setXtalParameters(index,param);
		    // the same index can be used for both EB and EE since there are always less than 68 trigger towers
		    ecaltpp.setXtalParameters(TCC,towerInTCC,stripInTower,xtalInStrip,param);
		  }
	  }
      
	}

	if (dataCard == "STRIP") {
	  infile[subdet]>>std::dec>>tccNb>>towerNbInTcc>>stripNbInTower ;
	  param.clear() ;
	  for (int i=0 ; i <NBstripparams[subdet] ; i++) {
	    infile[subdet]>>std::hex>>data ;
	    param.push_back(data) ;
	  }
	  std::vector<int> range = getRange(subdet, tccNb, towerNbInTcc, stripNbInTower) ;
	  for (int TCC = range[0] ; TCC < range[1] ; TCC++)
	    for (int towerInTCC = range[2] ; towerInTCC < range[3] ; towerInTCC++)
	      for (int stripInTower = range[4] ; stripInTower < range[5] ;  stripInTower++) {
		//	      int index = (68*TCC + towerInTCC)*5 + stripInTower ;
		//	      stripParam_[index] = param ;
		//		ecaltpp.setStripParameters(index,param);
		ecaltpp.setStripParameters(TCC,towerInTCC,stripInTower,param);
	      }
	  if (subdet!=0) { // repeat for EE eta<0
	    std::vector<int> range = getRange(-subdet, tccNb, towerNbInTcc, stripNbInTower) ;
	    for (int TCC = range[0] ; TCC < range[1] ; TCC++)
	      for (int towerInTCC = range[2] ; towerInTCC < range[3] ; towerInTCC++)
		for (int stripInTower = range[4] ; stripInTower < range[5] ;  stripInTower++) {
		  //	      int index = (68*TCC + towerInTCC)*5 + stripInTower ;
		  //	      stripParam_[index] = param ;
		  //		ecaltpp.setStripParameters(index,param);
		  ecaltpp.setStripParameters(TCC,towerInTCC,stripInTower,param);
		}
	  }
	}

	if (dataCard == "TOWER") {
	  infile[subdet]>>std::dec>>tccNb>>towerNbInTcc ;
	  param.clear() ;
	  for (int i=0 ; i <NBtowerparams[subdet] ; i++) {
	    infile[subdet]>>std::hex>>data ;
	    param.push_back(data) ;
	  }	
	  std::vector<int> range = getRange(subdet, tccNb, towerNbInTcc) ;
	  for (int TCC = range[0] ; TCC < range[1] ; TCC++)
	    for (int towerInTCC = range[2] ; towerInTCC < range[3] ; towerInTCC++) {
	      //int index = 68*TCC + towerInTCC ;	          
	      //	    towerParam_[index] = param ;
	      //		ecaltpp.setTowerParameters(index,param);
	      ecaltpp.setTowerParameters(TCC,towerInTCC,param);
	    }
      
	  if (subdet!=0) { // repeat for EE eta<0
	    std::vector<int> range = getRange(-subdet, tccNb, towerNbInTcc) ;
	    for (int TCC = range[0] ; TCC < range[1] ; TCC++)
	      for (int towerInTCC = range[2] ; towerInTCC < range[3] ; towerInTCC++) {
		//int index = 68*TCC + towerInTCC ;	          
		//	    towerParam_[index] = param ;
		//		ecaltpp.setTowerParameters(index,param);
		ecaltpp.setTowerParameters(TCC,towerInTCC,param);
	      }
	  }
	}
      
      }
      infile[subdet].close();
    }
  }
}

std::vector<int> EcalTrigPrimESProducer::getRange(int subdet, int tccNb, int towerNbInTcc, int stripNbInTower, int xtalNbInStrip)
{
  std::vector<int> range ;
  if (subdet == 0) { 
    // Barrel
    range.push_back(MIN_TCC_EB)  ; // stccNbMin
    range.push_back(MAX_TCC_EB) ; // tccNbMax
    range.push_back(MIN_TT_EB)  ; // towerNbMin
    range.push_back(MAX_TT_EB) ; // towerNbMax
    range.push_back(MIN_STRIP_EB)  ; // stripNbMin
    range.push_back(MAX_STRIP_EB)  ; // stripNbMax
    range.push_back(MIN_XTAL_EB)  ; // xtalNbMin
    range.push_back(MAX_XTAL_EB)  ; // xtalNbMax
  } else {
    // Endcap eta >0
    if (subdet >0 ) {
      range.push_back(MIN_TCC_EE_PLUS) ; // tccNbMin
      range.push_back(MAX_TCC_EE_PLUS) ; // tccNbMax
    } else { //endcap eta <0
      range.push_back(MIN_TCC_EE_MINUS) ; // tccNbMin
      range.push_back(MAX_TCC_EE_MINUS) ; // tccNbMax
    }
    range.push_back(MIN_TT_EE)  ; // towerNbMin
    range.push_back(MAX_TT_EE) ; // towerNbMax
    range.push_back(MIN_STRIP_EE)  ; // stripNbMin
    range.push_back(MAX_STRIP_EE)  ; // stripNbMax
    range.push_back(MIN_XTAL_EE)  ; // xtalNbMin
    range.push_back(MAX_XTAL_EE)  ; // xtalNbMax
  }

  if (tccNb>0) {
    range[0] = tccNb ; 
    range[1] = tccNb+1 ;
  }
  if (towerNbInTcc>0) {
    range[2] = towerNbInTcc ; 
    range[3] = towerNbInTcc+1 ;
  }
  if (stripNbInTower>0) {
    range[4] = stripNbInTower ; 
    range[5] = stripNbInTower+1 ;
  }
  if (xtalNbInStrip>0) {
    range[6] = xtalNbInStrip ; 
    range[7] = xtalNbInStrip+1 ;
  }

  return range ;
}
 

