#ifndef HPDLibrary_HPDNoiseData_h
#define HPDLibrary_HPDNoiseData_h

// --------------------------------------------------------
// Object to store correlated noise data for one HPD 
// Project: HPD noise library
// Author: F.Ratnikov UMd, Jan. 15, 2008
// $Id: HPDNoiseData.h,v 1.2 2008/01/16 20:49:09 fedor Exp $
// --------------------------------------------------------

#include "TObject.h"

#include <iostream>
#include <stdint.h>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseDataFrame.h"
#include <vector>

class HPDNoiseData {
 public:
  HPDNoiseData () {}
  virtual ~HPDNoiseData ();

  /// number of noise channels in the event
  unsigned size () const {return mData.size();}
  /// add another noise channel to the event
  void addChannel (HcalDetId fId, const float* fCharges);
  /// all channels contributing to the event
  std::vector<HcalDetId> getAllDetIds () const;
  /// retrive frame for the given index
  const HPDNoiseDataFrame& getDataFrame (size_t i) const;
  /// reset event to empty state
  void clear () {mData.clear ();}
 private:
  std::vector<HPDNoiseDataFrame> mData;

  ClassDef(HPDNoiseData,1)
};

/// printout
std::ostream& operator<< (std::ostream&, const HPDNoiseData&);

#endif
