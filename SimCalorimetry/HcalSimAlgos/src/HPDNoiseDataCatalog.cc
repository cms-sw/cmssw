// --------------------------------------------------------
// Object to store HPD instance name and noise rate for the instance
// Project: HPD noise library
// Author: F.Ratnikov UMd, Jan. 15, 2008
// $Id: BasicJet.h,v 1.11 2007/09/20 21:04:43 fedor Exp $
// --------------------------------------------------------

#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseDataCatalog.h"

HPDNoiseDataCatalog::~HPDNoiseDataCatalog () {}


std::ostream& operator<< (std::ostream& fStream, const HPDNoiseDataCatalog& fCatalog) {
  fStream << "Name:Rate";
  for (size_t i = 0; i < fCatalog.size(); ++i) fStream << ' ' << fCatalog.getName (i) << ':' << fCatalog.getRate (i);
  return fStream;
}
