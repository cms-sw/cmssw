#include <SimCalorimetry/EcalTrigPrimAlgos/interface/DBInterface.h>
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <iostream>
#include <fstream>
#include <string>


DBInterface::DBInterface(std::string dbFilenameEB,std::string dbFilenameEE) 
{
  int NBxtalparams  = 9 ;
  int NBstripparams = 6 ;
  int NBtowerparams = 1029 ;
  std::vector<unsigned int> param ;

  std::string filenameEB="SimCalorimetry/EcalTrigPrimProducers/data/"+dbFilenameEB;
//  edm::FileInPath fileInPath("SimCalorimetry/EcalTrigPrimProducers/data/TPG.txt");
  edm::FileInPath fileInPath(filenameEB);
  std::ifstream infile (fileInPath.fullPath().c_str()) ;
  if (infile.is_open()) {

    // xtal structure : ped, mult, shift [gain12] , ped, mult ,shift [gain6], ... [gain1]
    // strip structure : sliding , weight1, weight2, weight3, weight4, weight5
    // tower structure : lut[0], ... , lut[1023], el, eh, tl, th, lut_fg

    unsigned int data ;
    std::string dataCard ;
    int smNb, towerNbInSm, stripNbInTower, xtalNbInStrip ;

    while (!infile.eof()) {
      infile>>dataCard ;

      if (dataCard == "CRYSTAL") {
	infile>>std::dec>>smNb>>towerNbInSm>>stripNbInTower>>xtalNbInStrip ;
	param.clear() ;
	for (int i=0 ; i <NBxtalparams ; i++) {
	  infile>>std::hex>>data ;
	  param.push_back(data) ;
	}
	std::vector<int> range = getRange(smNb, towerNbInSm, stripNbInTower, xtalNbInStrip) ;
	for (int SM = range[0] ; SM < range[1] ; SM++)
	  for (int towerInSM = range[2] ; towerInSM < range[3] ; towerInSM++)
	    for (int stripInTower = range[4] ; stripInTower < range[5] ;  stripInTower++)
	      for (int xtalInStrip =  range[6] ; xtalInStrip <  range[7] ; xtalInStrip++) {
		int index = ((68*SM + towerInSM)*5 + stripInTower)*5 +  xtalInStrip ;		
		xtalParam_[index] = param ;
	      }
      }

      if (dataCard == "STRIP") {
	infile>>std::dec>>smNb>>towerNbInSm>>stripNbInTower ;
	param.clear() ;
	for (int i=0 ; i <NBstripparams ; i++) {
	  infile>>std::hex>>data ;
	  param.push_back(data) ;
	}
	std::vector<int> range = getRange(smNb, towerNbInSm, stripNbInTower) ;
	for (int SM = range[0] ; SM < range[1] ; SM++)
	  for (int towerInSM = range[2] ; towerInSM < range[3] ; towerInSM++)
	    for (int stripInTower = range[4] ; stripInTower < range[5] ;  stripInTower++) {
	      int index = (68*SM + towerInSM)*5 + stripInTower ;
	      stripParam_[index] = param ;
	    }
      }

      if (dataCard == "TOWER") {
	infile>>std::dec>>smNb>>towerNbInSm ;
	param.clear() ;
	for (int i=0 ; i <NBtowerparams ; i++) {
	  infile>>std::hex>>data ;
	  param.push_back(data) ;
	}	
	std::vector<int> range = getRange(smNb, towerNbInSm) ;
	for (int SM = range[0] ; SM < range[1] ; SM++)
	  for (int towerInSM = range[2] ; towerInSM < range[3] ; towerInSM++) {
	    int index = 68*SM + towerInSM ;	          
	    towerParam_[index] = param ;
	  }
      }

    }
  }
  infile.close();
}

std::vector<unsigned int> DBInterface::getTowerParameters(int SM, int towerInSM, bool print)
{
  // SM = 1->36 , towerInSM = 1->68
  int index = 68*SM + towerInSM ;
  std::map<int, std::vector<unsigned int> >::iterator it ;
  it = towerParam_.find(index) ;
  if (it == towerParam_.end()) {
    std::cout<<"EXCEPTION DBInterface::getTowerParameters"<<std::endl ;//PP should throw exception!!!
    std::cout<<"===> DBInterface::getTowerParameters("<<std::dec<<SM<<", "<<towerInSM<<")"<<std::endl ;
  }
  std::vector<unsigned int> param = it->second ;
  if (print) {
    std::cout<<"===> DBInterface::getTowerParameters("<<std::dec<<SM<<", "<<towerInSM<<")"<<std::endl ;
    for (int i=0 ; i<1024 ; i++) std::cout<<"LUT["<<std::dec<<i<<"] = "<<std::hex<<param[i]<<std::endl ;
    std::cout<<"Fine Grain:  el="<<param[1024]<<", eh="<<param[1025]
	     <<", tl="<<param[1026]<<",  th="<<param[1027]
	     <<", lut_fg="<<param[1028]<<std::endl ;
  }
  return param ;
}

std::vector<unsigned int> DBInterface::getStripParameters(int SM, int towerInSM, int stripInTower, bool print)
{
  // SM = 1->36 , towerInSM = 1->68, stripInTower = 1->5
  int index = (68*SM + towerInSM)*5 + stripInTower ;
  std::map<int, std::vector<unsigned int> >::const_iterator it ;
  it = stripParam_.find(index) ;
  if (it == stripParam_.end()) {
    std::cout<<"EXCEPTION DBInterface::getStripParameters"<<std::endl ;//PP should throw exception!!!
    std::cout<<"===> DBInterface::getStripParameters("<<std::dec<<SM<<", "<<towerInSM<<", "<<stripInTower<<")"<<std::endl ;
  }
  std::vector<unsigned int> param = it->second ;
  if (print) {
    std::cout<<"===> DBInterface::getStripParameters("<<std::dec<<SM<<", "<<towerInSM<<", "<<stripInTower<<")"<<std::endl ;
    std::cout<<"sliding window = "<<std::hex<<param[0]<<std::endl ;
    for (int i=0 ; i<5 ; i++) std::cout<<"Weight["<<std::dec<<i<<"] ="<<std::hex<<param[i+1]<<std::endl ;
  }
  return param ;
}

std::vector<unsigned int> DBInterface::getXtalParameters(int SM, int towerInSM, int stripInTower, int xtalInStrip, bool print)
{
  // SM = 1->36 , towerInSM = 1->68, stripInTower = 1->5, xtalInStrip = 1->5
  int index = ((68*SM + towerInSM)*5 + stripInTower)*5 +  xtalInStrip ;
  std::map<int, std::vector<unsigned int> >::const_iterator it ;
  it = xtalParam_.find(index) ;
  if (it == xtalParam_.end()) {
   std::cout<<"EXCEPTION DBInterface::getXtalParameters"<<std::endl ;//PP should throw exception!!!
   std::cout<<"===> DBInterface::getXtalParameters("<<std::dec<<SM<<", "
	     <<towerInSM<<", "<<stripInTower<<", "<<xtalInStrip<<")"<<std::endl ;
  }
  std::vector<unsigned int> param = it->second ;
  if (print) {
    std::cout<<"===> DBInterface::getXtalParameters("<<std::dec<<SM<<", "
	     <<towerInSM<<", "<<stripInTower<<", "<<xtalInStrip<<")"<<std::endl ;
    std::cout<<"Gain12, ped = "<<std::hex<<param[0]<<", mult = "<<param[1]<<", shift = "<<param[2]<<std::endl ;
    std::cout<<"Gain6,  ped = "<<std::hex<<param[3]<<", mult = "<<param[4]<<", shift = "<<param[5]<<std::endl ;
    std::cout<<"Gain1,  ped = "<<std::hex<<param[6]<<", mult = "<<param[7]<<", shift = "<<param[8]<<std::endl ;
  }
  return param ;
}


std::vector<int> DBInterface::getRange(int smNb, int towerNbInSm, int stripNbInTower, int xtalNbInStrip)
{
  std::vector<int> range ;
  range.push_back(1)  ; // smNbMin
  range.push_back(37) ; // smNbMax
  range.push_back(1)  ; // towerNbMin
  range.push_back(69) ; // towerNbMax
  range.push_back(1)  ; // stripNbMin
  range.push_back(6)  ; // stripNbMax
  range.push_back(1)  ; // xtalNbMin
  range.push_back(6)  ; // xtalNbMax

  if (smNb>0) {
    range[0] = smNb ; 
    range[1] = smNb+1 ;
  }
  if (towerNbInSm>0) {
    range[2] = towerNbInSm ; 
    range[3] = towerNbInSm+1 ;
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
