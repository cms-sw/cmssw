#include "SimCalorimetry/HcalSimAlgos/interface/HcalTDC.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Random/RandGaussQ.h"

HcalTDC::HcalTDC(double threshold_currentTDC)
    : theTDCParameters(), theDbService(nullptr), threshold_currentTDC_(threshold_currentTDC), lsb(3.74) {}

HcalTDC::~HcalTDC() {}

//template <class Digi>
void HcalTDC::timing(const CaloSamples& lf, QIE11DataFrame& digi) const {
  float const TDC_Threshold(getThreshold());
  bool risingReady(true);
  int tdcBins = theTDCParameters.nbins();
  bool hasTDCValues = true;
  if (lf.preciseSize() == 0)
    hasTDCValues = false;

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
    int TDC_RisingEdge = (risingReady) ? theTDCParameters.noTransitionCode() : theTDCParameters.alreadyTransitionCode();
    int preciseBegin = ibin * tdcBins;
    int preciseEnd = preciseBegin + tdcBins;

    if (hasTDCValues) {
      for (int i = preciseBegin; i < preciseEnd; ++i) {  //find the TDC time value in each TS

        if ((!risingReady) && (i == preciseBegin) && (i != 0)) {
          if (((lf.preciseAt(i + 1) - lf.preciseAt(i - 1)) > TDC_Threshold)) {
            TDC_RisingEdge = theTDCParameters.alreadyTransitionCode();
            break;
          } else
            risingReady = true;
        }

        if (risingReady) {
          if (i != 399 && i != 0 && (lf.preciseAt(i + 1) - lf.preciseAt(i - 1)) > TDC_Threshold) {
            risingReady = false;
            TDC_RisingEdge = i - preciseBegin;
          } else if (i == 0 && (lf.preciseAt(i + 1) - lf.preciseAt(i)) / 0.5 > TDC_Threshold) {
            risingReady = false;
            TDC_RisingEdge = i - preciseBegin;
          } else if (i == (preciseEnd - 1))
            TDC_RisingEdge = theTDCParameters.noTransitionCode();
        }

        if ((!risingReady) && (i == (preciseEnd - 1)) && (i != 399)) {
          if (((lf.preciseAt(i + 1) - lf.preciseAt(i - 1)) < TDC_Threshold)) {
            risingReady = true;
          }
        }
      }  //end of looping precise bins
    }

    // change packing to allow for special codes
    int packedTDC = TDC_RisingEdge;
    digi.setSample(ibin, digi[ibin].adc(), packedTDC, digi[ibin].soi());

  }  // loop over bunch crossing bins
}

void HcalTDC::setDbService(const HcalDbService* service) { theDbService = service; }
