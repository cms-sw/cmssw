// --------------------------------------------------------
// Object to store all timeslices of noise signal frame
// Project: HPD noise library
// Author: F.Ratnikov UMd, Jan. 15, 2008
// $Id: HPDNoiseDataFrame.cc,v 1.2 2008/01/16 17:36:00 fedor Exp $
// --------------------------------------------------------

#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseDataFrame.h"

HPDNoiseDataFrame::HPDNoiseDataFrame (HcalDetId fId, const float* fCharges) 
  : mId (fId.rawId())
{
  for (size_t i = 0; i < FRAMESIZE; ++i) mCharge[i] = fCharges[i];
}

HPDNoiseDataFrame::~HPDNoiseDataFrame () {}

std::ostream& operator<< (std::ostream& fStream, const HPDNoiseDataFrame& fFrame) {
  fStream << fFrame.id();
  for (size_t i = 0; i < FRAMESIZE; ++i) fStream << ' ' << i << ':' << fFrame.charge (i);
  return fStream;
}
