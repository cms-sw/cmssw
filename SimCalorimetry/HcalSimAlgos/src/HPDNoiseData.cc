// --------------------------------------------------------
// Object to store correlated noise data for one HPD 
// Project: HPD noise library
// Author: F.Ratnikov UMd, Jan. 15, 2008
// $Id: BasicJet.h,v 1.11 2007/09/20 21:04:43 fedor Exp $
// --------------------------------------------------------

#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseData.h"

HPDNoiseData::~HPDNoiseData () {}

void HPDNoiseData::addChannel (HcalDetId fId, const float* fCharges) {
  mData.push_back (HPDNoiseDataFrame (fId, fCharges));
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

