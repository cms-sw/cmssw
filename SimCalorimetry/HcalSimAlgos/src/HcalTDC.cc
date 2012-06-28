#include "SimCalorimetry/HcalSimAlgos/src/HcalTDC.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

HcalTDC::HcalTDC() :
  theTDCParameters(),
  theDbService(0),
  theRandGaussQ(0)
{
}

HcalTDC::~HcalTDC()
{
  delete theRandGaussQ;
}

void HcalTDC::timing(const CaloSamples & lf, HcalUpgradeDataFrame & digi) const
{
  float TDC_Threshold = getThreshold(digi.id());
  bool alreadyOn = false;
  int tdcBins = theTDCParameters.nbins();
  // start with a loop over 10 samples
  bool hasTDCValues=true;
  if (lf.preciseSize()==0 ) hasTDCValues=false;
  for(int ibin = 0; ibin < lf.size(); ++ibin) {
    /*
    If in a given 25ns bunch/time sample, the pulse is above TDC_Thresh  
    already, then TDC_RisingEdge=0
    and then one can still have a TDC_FallingEdge that is non-zero.  If  
    the pulse never falls below threshold having
    started above threshold, then TDC_FallingEdge=31.  If the pulse never  
    went above threshold, then
    TDC_RisingEdge=31 and TDC_FallingEdge=0.
    */
    int TDC_RisingEdge = alreadyOn ? 0 : tdcBins-1;
    int TDC_FallingEdge = alreadyOn ? tdcBins-1 : 0;
    int preciseBegin = ibin * tdcBins;
    int preciseEnd = preciseBegin + tdcBins;
    if ( hasTDCValues) {
      for(int i = preciseBegin; i < preciseEnd; ++i)
	{ 
	  if(alreadyOn)
	    {
	      if(lf.preciseAt(i) < TDC_Threshold)
		{
		  alreadyOn = false;
		  TDC_FallingEdge = i-preciseBegin;
		}
	    }
	  else 
	    {
	      if(lf.preciseAt(i) > TDC_Threshold)
		{
		  alreadyOn = true;
		  TDC_RisingEdge = i-preciseBegin;
		  // the flag for hasn't gone low yet
		  TDC_FallingEdge = tdcBins-1;
		}
	    }
	}
    }
    int packedTDC = TDC_RisingEdge + tdcBins * TDC_FallingEdge;
    digi.setSample(ibin, digi.adc(ibin), packedTDC, true);
  } // loop over bunch crossing bins
}

double HcalTDC::getThreshold(const HcalGenericDetId & detId) const {
  // subtract off pedestal and noise once
  double pedestal = theDbService->getHcalCalibrations(detId).pedestal(0);
  double pedestalWidth = theDbService->getHcalCalibrationWidths(detId).pedestal(0);
  return 100. - theRandGaussQ->shoot(pedestal,  pedestalWidth);
}

void HcalTDC::setRandomEngine(CLHEP::HepRandomEngine & engine)
{
  theRandGaussQ = new CLHEP::RandGaussQ(engine);
}

void HcalTDC::setDbService(const HcalDbService * service) {
  theDbService = service;
}

