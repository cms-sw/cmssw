#ifndef HPDLibrary_HPDNoiseData_h
#define HPDLibrary_HPDNoiseData_h

// --------------------------------------------------------
// Object to store correlated noise data for one HPD 
// Project: HPD noise library
// Author: F.Ratnikov UMd, Jan. 15, 2008
// $Id: HPDNoiseData.h,v 1.4 2008/08/04 22:07:08 fedor Exp $
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
  /// class name
  static const char* className () {return "HPDNoiseData";}
  /// branch name
  static const char* branchName () {return "data";}
 private:
  std::vector<HPDNoiseDataFrame> mData;

};

/// printout
std::ostream& operator<< (std::ostream&, const HPDNoiseData&);

#endif
