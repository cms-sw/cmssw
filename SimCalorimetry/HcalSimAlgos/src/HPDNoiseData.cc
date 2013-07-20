// --------------------------------------------------------
// Object to store correlated noise data for one HPD 
// Project: HPD noise library
// Author: F.Ratnikov UMd, Jan. 15, 2008
// $Id: HPDNoiseData.cc,v 1.2 2012/06/07 18:12:43 wmtan Exp $
// --------------------------------------------------------

#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseData.h"

HPDNoiseData::~HPDNoiseData () {}

void HPDNoiseData::addChannel (HcalDetId fId, const float* fCharges) {
  mData.emplace_back (fId, fCharges);
}

std::vector<HcalDetId> HPDNoiseData::getAllDetIds () const {
  std::vector<HcalDetId> result;
  for (size_t i = 0; i < mData.size(); ++i) result.push_back (getDataFrame(i).id());
  return result;
}

const HPDNoiseDataFrame& HPDNoiseData::getDataFrame (size_t i) const {
  return mData[i];
}

std::ostream& operator<< (std::ostream& fStream, const HPDNoiseData& fData) {
  for (size_t i = 0; i < fData.size (); ++i) fStream << fData.getDataFrame (i) << std::endl;
  return fStream;
}

