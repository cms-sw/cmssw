#include <SimCalorimetry/EcalTrigPrimAlgos/interface/DBInterface.h>
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <iostream>
#include <fstream>


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

    unsigned int data ;
    // xtal structure : ped, mult, shift [gain12] , ped, mult ,shift [gain6], ... [gain1]
    param.clear() ;
    for (int i=0 ; i <NBxtalparams ; i++) {
      infile>>std::hex>>data ;
      param.push_back(data) ;
    }
    xtalParam_[0] = param ;

    // strip structure : sliding , weight1, weight2, weight3, weight4, weight5
    param.clear() ;
    for (int i=0 ; i <NBstripparams ; i++) {
      infile>>std::hex>>data ;
      param.push_back(data) ;
    }
    stripParam_[0] = param ;

    // tower structure : lut[0], ... , lut[1023], el, eh, tl, th, lut_fg
    param.clear() ;
    for (int i=0 ; i <NBtowerparams ; i++) {
      infile>>std::hex>>data ;
      param.push_back(data) ;
    }
    towerParam_[0] = param ;

  }
  infile.close();
}

std::vector<unsigned int> DBInterface::getTowerParameters(int SM, int towerInSM)
{
  // SM = 1->36 , towerInSM = 1->68
  int index = 68*SM + towerInSM ;
  std::map<int, std::vector<unsigned int> >::iterator it ;
  it = towerParam_.find(index) ;
  if (it == towerParam_.end()) it = towerParam_.find(0) ; // we use the default
  std::vector<unsigned int> param = it->second ;
  return param ;
}

std::vector<unsigned int> DBInterface::getStripParameters(int SM, int towerInSM, int stripInTower)
{
  // SM = 1->36 , towerInSM = 1->68, stripInTower = 1->5
  int index = (68*SM + towerInSM)*5 + stripInTower ;
  std::map<int, std::vector<unsigned int> >::const_iterator it ;
  it = stripParam_.find(index) ;
  if (it == stripParam_.end()) it = stripParam_.find(0) ; // we use the default
  std::vector<unsigned int> param = it->second ;
  return param ;
}

std::vector<unsigned int> DBInterface::getXtalParameters(int SM, int towerInSM, int stripInTower, int xtalInStrip)
{
  // SM = 1->36 , towerInSM = 1->68, stripInTower = 1->5, xtalInStrip = 1->5
  int index = ((68*SM + towerInSM)*5 + stripInTower)*5 +  xtalInStrip ;
  std::map<int, std::vector<unsigned int> >::const_iterator it ;
  it = xtalParam_.find(index) ;
  if (it == xtalParam_.end()) it = xtalParam_.find(0) ; // we use the default
  std::vector<unsigned int> param = it->second ;
  return param ;
}


