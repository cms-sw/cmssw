#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"  
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
  theHOZecotekSiPMParameters( 4000., 3.0, // 1 mip = 15 pe = 45 fC
                   217., 5,
                   10, 5, true, true,
                   1, std::vector<double>(16, 217.)),
  theHOHamamatsuSiPMParameters( 4000., 3.0,
                   217., 5,
                   10, 5, true, true,
                   1, std::vector<double>(16, 217.)),
  theHFParameters1(6., 2.79,
		   1/0.278 , -4,
		   true),
  theHFParameters2(6., 2.06,
		   1/0.267 , -4,
		   true),
  theZDCParameters(1., 4.3333,
		   2.09 , -4,
		   false),
  theHOZecotekDetIds(),
  theHOHamamatsuDetIds()
{
  theHOZecotekSiPMParameters.thePixels = 36000;
  theHOHamamatsuSiPMParameters.thePixels = 960;
}
/*
  CaloSimParameters(double simHitToPhotoelectrons, double photoelectronsToAnalog,
                 double samplingFactor, double timePhase,
                 int readoutFrameSize, int binOfMaximum,
                 bool doPhotostatistics, bool syncPhase=true);

*/

HcalSimParameterMap::HcalSimParameterMap(const edm::ParameterSet & p)
: theHBParameters(  p.getParameter<edm::ParameterSet>("hb") ),
  theHEParameters(  p.getParameter<edm::ParameterSet>("he") ),
  theHOParameters(  p.getParameter<edm::ParameterSet>("ho") ),
  theHOZecotekSiPMParameters(  p.getParameter<edm::ParameterSet>("hoZecotek") ),
  theHOHamamatsuSiPMParameters(  p.getParameter<edm::ParameterSet>("hoHamamatsu") ),
  theHFParameters1( p.getParameter<edm::ParameterSet>("hf1") ),
  theHFParameters2( p.getParameter<edm::ParameterSet>("hf2") ),
  theZDCParameters( p.getParameter<edm::ParameterSet>("zdc") )
{
}

const CaloSimParameters & HcalSimParameterMap::simParameters(const DetId & detId) const {
  HcalGenericDetId genericId(detId);
  if(genericId.isHcalZDCDetId())
    return theZDCParameters;
  HcalDetId hcalDetId(detId);
  if(hcalDetId.subdet() == HcalBarrel) {
     return theHBParameters;
  } else if(hcalDetId.subdet() == HcalEndcap) {
     return theHEParameters;
  } else if(hcalDetId.subdet() == HcalOuter) {
     if(std::find(theHOZecotekDetIds.begin(),
        theHOZecotekDetIds.end(), hcalDetId) != theHOZecotekDetIds.end())
     {
       return theHOZecotekSiPMParameters;
     }
     if(std::find(theHOHamamatsuDetIds.begin(),
        theHOHamamatsuDetIds.end(), hcalDetId) != theHOHamamatsuDetIds.end())
     {
       return theHOHamamatsuSiPMParameters;
     }
     else
     {
       return theHOParameters;
     }
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
  theHOZecotekSiPMParameters.setDbService(dbService);
  theHOHamamatsuSiPMParameters.setDbService(dbService);
  theHFParameters1.setDbService(dbService);
  theHFParameters2.setDbService(dbService);
  theZDCParameters.setDbService(dbService);
}

void HcalSimParameterMap::setFrameSize(const DetId & detId, int frameSize)
{
  HcalGenericDetId genericId(detId);
  if(genericId.isHcalZDCDetId())
    setFrameSize(theZDCParameters, frameSize);
  else
  {
    HcalDetId hcalDetId(detId);
    if(hcalDetId.subdet() == HcalForward) {
      // do both depths
      setFrameSize(theHFParameters1,frameSize);
      setFrameSize(theHFParameters2,frameSize);
    }
    else
    {
      CaloSimParameters & parameters = const_cast<CaloSimParameters &>(simParameters(detId));
      setFrameSize(parameters, frameSize);
    }
  }
}


void HcalSimParameterMap::setFrameSize(CaloSimParameters & parameters, int frameSize)
{
  int binOfMaximum = 5;
  if(frameSize == 10) {}
  else if(frameSize == 6) binOfMaximum = 4;
  else {
    edm::LogError("HcalSimParameterMap")<< "Bad HCAL frame size " << frameSize;
  }
  if(parameters.readoutFrameSize() != frameSize)
  {
    edm::LogWarning("HcalSignalGenerator")<< "Mismatch in frame sizes.  Setting to " << frameSize;
    parameters.setReadoutFrameSize(frameSize);
    parameters.setBinOfMaximum(binOfMaximum);
  }
}
 
