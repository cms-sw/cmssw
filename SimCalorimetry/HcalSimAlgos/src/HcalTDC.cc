#include "SimCalorimetry/HcalSimAlgos/interface/HcalTDC.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalTDC::HcalTDC(unsigned int thresholdDAC) : theTDCParameters(), 
					      theDbService(0),
					      theDAC(thresholdDAC),
					      lsb(3.74),
					      theRandGaussQ(0) {}

HcalTDC::~HcalTDC() {
  if (theRandGaussQ) delete theRandGaussQ;
}

void HcalTDC::timing(const CaloSamples& lf, HcalUpgradeDataFrame& digi) const {

  float const TDC_Threshold(getThreshold(digi.id()));
  float const TDC_Threshold_hyst(TDC_Threshold);
  bool risingReady(true), fallingReady(false);
  int tdcBins = theTDCParameters.nbins();
  // start with a loop over 10 samples
  bool hasTDCValues=true;
  if (lf.preciseSize()==0 ) hasTDCValues=false;

  // "AC" hysteresis from Tom:  There cannot be a second transition
  // within the QIE within 3ns
  int const rising_ac_hysteresis(5);
  int lastRisingEdge(0);

  // if (hasTDCValues)
  //   std::cout << digi.id() 
  // 	      << " threshold: " << TDC_Threshold
  // 	      << '\n';
  for (int ibin = 0; ibin < lf.size(); ++ibin) {
    /*
    If in a given 25ns bunch/time sample, the pulse is above
    TDC_Thresh already, then TDC_RisingEdge=0 if it was low in the
    last precision bin on the previous bunch crossing, otherwise,
    TDC_RisingEdge=63 if the pulse never crosses the threshold
    having started off, then the special code is 62 and then
    one can still have a TDC_FallingEdge that is valid.  If the pulse
    never falls below threshold having started above threshold (or
    goes above threshold in the bunch crossing and doesn't come down),
    then TDC_FallingEdge=.  If the pulse never went above
    threshold, then TDC_RisingEdge=63 and
    TDC_FallingEdge=62.
    */
    // special codes
    int TDC_RisingEdge = (risingReady) ? 
      theTDCParameters.noTransitionCode() :
      theTDCParameters.alreadyTransitionCode();
    int TDC_FallingEdge = (fallingReady) ? 
      theTDCParameters.alreadyTransitionCode() :
      theTDCParameters.noTransitionCode();
    int preciseBegin = ibin * tdcBins;
    int preciseEnd = preciseBegin + tdcBins;
    if ( hasTDCValues) {
      // std::cout << " alreadyOn: " << alreadyOn << '\n';
      for(int i = preciseBegin; i < preciseEnd; ++i) { 
      // 	std::cout << " preciseBin: " << i
      // 			    << " preciseAt(i): " << lf.preciseAt(i);
	if( (fallingReady) && (i%3 == 0) && (lf.preciseAt(i) < TDC_Threshold)) {
	  fallingReady = false;
	  TDC_FallingEdge = i-preciseBegin;
	  // std::cout << " falling ";
	}
	if ((risingReady) && (lf.preciseAt(i) > TDC_Threshold)) {
	  risingReady = false;
	  TDC_RisingEdge = i-preciseBegin;
	  lastRisingEdge = 0;
	  // std::cout << " rising ";
	}
	// std::cout << '\n';

	//This is the place for hysteresis code.  Need to to arm the
	//tdc by setting the risingReady or fallingReady to true.

	if ((!fallingReady) && (lf.preciseAt(i) > TDC_Threshold)) {
	  fallingReady = true;
	  if (TDC_FallingEdge == theTDCParameters.alreadyTransitionCode())
	    TDC_FallingEdge = theTDCParameters.noTransitionCode();
	}
	if ((!risingReady) && (lastRisingEdge > rising_ac_hysteresis) &&
	    (lf.preciseAt(i) < TDC_Threshold_hyst)) {
	  risingReady = true;
	  if (TDC_RisingEdge == theTDCParameters.alreadyTransitionCode())
	    TDC_RisingEdge = theTDCParameters.noTransitionCode();
	}
	++lastRisingEdge;
      }
    }
    // change packing to allow for special codes
    int packedTDC = TDC_RisingEdge + (tdcBins*2) * TDC_FallingEdge;
    digi.setSample(ibin, digi.adc(ibin), packedTDC, true);
    // if ( hasTDCValues) {
    //   std::cout << " sample: " << ibin 
    // 		<< " adc: " << digi.adc(ibin)
    // 		<< " fC: " << digi[ibin].nominal_fC()
    // 		<< " risingEdge: " << TDC_RisingEdge
    // 		<< " fallingEdge: " << TDC_FallingEdge
    // 		<< " packedTDC: " << packedTDC
    // 		<< std::endl;
    // }
  } // loop over bunch crossing bins
}

double HcalTDC::getThreshold(const HcalGenericDetId & detId) const {
  // subtract off pedestal and noise once
  double pedestal = theDbService->getHcalCalibrations(detId).pedestal(0);
  double pedestalWidth = theDbService->getHcalCalibrationWidths(detId).pedestal(0);
  // here the TDCthreshold_ is a multiple of the least significant bit 
  // for the TDC portion of the QIE.  The nominal reference is 40 uA which is
  // divided by an 8 bit DAC.  This means the least significant bit is 0.135 uA
  // in the TDC circuit or 3.74 uA at the input current.

  // the pedestal is assumed to be evenly distributed in time with some
  // random variation.

  return lsb*theDAC - theRandGaussQ->shoot(pedestal,  pedestalWidth)/theTDCParameters.deltaT()/theTDCParameters.nbins();
}

double HcalTDC::getHysteresisThreshold(double nominal) const {
  // the TDC will "re-arm" when it crosses the threshold minus 2.5 mV.
  // the lsb is 44kOhm * (3.74 uA / 24) = 6.86 mV.  2.5 mV is 0.36 lsb.
  // with the this means that the hysteresis it isn't far from the nominal 
  // threshold

  return nominal - 0.365*lsb;
}

void HcalTDC::setRandomEngine(CLHEP::HepRandomEngine & engine) {
  theRandGaussQ = new CLHEP::RandGaussQ(engine);
}

void HcalTDC::setDbService(const HcalDbService * service) {
  theDbService = service;
}
