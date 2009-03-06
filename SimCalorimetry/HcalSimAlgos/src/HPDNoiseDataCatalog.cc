// --------------------------------------------------------
// Object to store HPD instance name and noise rate for the instance
// Project: HPD noise library
// Author: F.Ratnikov UMd, Jan. 15, 2008
// $Id: HPDNoiseDataCatalog.cc,v 1.1 2008/01/16 02:12:40 fedor Exp $
// --------------------------------------------------------

#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseDataCatalog.h"

HPDNoiseDataCatalog::~HPDNoiseDataCatalog () {}

void HPDNoiseDataCatalog::setRate (const std::string& fName, float fRate) {
  for (size_t i = 0; i < mHpdName.size(); ++i) {
    if (fName == mHpdName[i]) mRate[i] = fRate;
  }
}


std::ostream& operator<< (std::ostream& fStream, const HPDNoiseDataCatalog& fCatalog) {
  fStream << "Name:Rate";
  for (size_t i = 0; i < fCatalog.size(); ++i) fStream << ' ' << fCatalog.getName (i) << ':' << fCatalog.getRate (i);
  return fStream;
}
