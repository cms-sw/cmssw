#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
  

HcalSimParameterMap::HcalSimParameterMap() :
  theHBParameters(2000., 0.3305,
                   117, 5, 
                   10, 5, true, true,
                   1, std::vector<double>(16, 117.)),
  theHEParameters(2000., 0.3305,
                   178, 5,
                   10, 5, true, true,
                   16, std::vector<double>(16, 178.)),
  theHOParameters( 4000., 0.3065, 
                   217., 5, 
                   10, 5, true, true,
                   1, std::vector<double>(16, 217.)),
  theHFParameters1(1., 5.917,
                 2.84 , -4,
                6, 4, false),
  theHFParameters2(1., 4.354,
                 2.09 , -4,
                6, 4, false)
{
}
/*
  CaloSimParameters(double photomultiplierGain, double amplifierGain,
                 double samplingFactor, double timePhase,
                 int readoutFrameSize, int binOfMaximum,
                 bool doPhotostatistics)
*/

HcalSimParameterMap::HcalSimParameterMap(const edm::ParameterSet & p)
: theHBParameters(  p.getParameter<edm::ParameterSet>("hb") ),
  theHEParameters(  p.getParameter<edm::ParameterSet>("he") ),
  theHOParameters(  p.getParameter<edm::ParameterSet>("ho") ),
  theHFParameters1( p.getParameter<edm::ParameterSet>("hf1") ),
  theHFParameters2( p.getParameter<edm::ParameterSet>("hf2") )
{
}


const CaloSimParameters & HcalSimParameterMap::simParameters(const DetId & detId) const {
  HcalDetId hcalDetId(detId);
  if(hcalDetId.subdet() == HcalBarrel) {
     return theHBParameters;
  } else if(hcalDetId.subdet() == HcalEndcap) {
     return theHEParameters;
  } else if(hcalDetId.subdet() == HcalOuter) {
     return theHOParameters;
  } else { // HF
    if(hcalDetId.depth() == 1) {
      return theHFParameters1;
    } else {
      return theHFParameters2;
    }
  }
}


void HcalSimParameterMap::setDbService(const HcalDbService * dbService)
{
  theHBParameters.setDbService(dbService);
  theHEParameters.setDbService(dbService);
  theHOParameters.setDbService(dbService);
}
