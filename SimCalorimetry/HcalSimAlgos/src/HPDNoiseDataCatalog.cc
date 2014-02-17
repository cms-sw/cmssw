// --------------------------------------------------------
// Object to store HPD instance name and noise rate for the instance
// Project: HPD noise library
// Author: F.Ratnikov UMd, Jan. 15, 2008
// $Id: HPDNoiseDataCatalog.cc,v 1.3 2008/07/21 18:30:03 tyetkin Exp $
// --------------------------------------------------------

#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseDataCatalog.h"

HPDNoiseDataCatalog::~HPDNoiseDataCatalog () {}

void  HPDNoiseDataCatalog::setRate (const std::string& fName, float fDischargeRate, 
                                      float fIonFeedbackFirstPeakRate, float fIonFeedbackSecondPeakRate,
				      float fElectronEmissionRate){
  for (size_t i = 0; i < mHpdName.size(); ++i) {
    if (fName == mHpdName[i]){
        mDischargeRate[i] = fDischargeRate;
        mIonFeedbackFirstPeakRate[i] = fIonFeedbackFirstPeakRate;
        mIonFeedbackSecondPeakRate[i] = fIonFeedbackSecondPeakRate;
        mElectronEmissionRate[i] = fElectronEmissionRate;
    }
  }
}


std::ostream& operator<< (std::ostream& fStream, const HPDNoiseDataCatalog& fCatalog) {
  fStream << "Name:DischargeRate:IonFeedbackRate:ElectronEmissionRate";
  for (size_t i = 0; i < fCatalog.size(); ++i) fStream << ' ' << fCatalog.getName (i) << ':' 
       << fCatalog.getDischargeRate(i) << ':' 
       << fCatalog.getIonFeedbackFirstPeakRate(i) << ':' << fCatalog.getIonFeedbackSecondPeakRate(i) << ':'
       << fCatalog.getElectronEmissionRate(i);
  return fStream;
}
