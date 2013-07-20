#ifndef HPDLibrary_HPDNoiseDataFrame_h
#define HPDLibrary_HPDNoiseDataFrame_h

// --------------------------------------------------------
// Object to store all timeslices of noise signal frame
// Project: HPD noise library
// Author: F.Ratnikov UMd, Jan. 15, 2008
// $Id: HPDNoiseDataFrame.h,v 1.2 2008/08/04 22:07:08 fedor Exp $
// --------------------------------------------------------

#include "TObject.h"

#include <iostream>
#include <stdint.h>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"


namespace {
  const unsigned int FRAMESIZE = 10;
}

class HPDNoiseDataFrame {
 public:
  HPDNoiseDataFrame () {};
  HPDNoiseDataFrame (HcalDetId fId, const float* fCharges);
  virtual ~HPDNoiseDataFrame ();
  /// detId for the frame
  HcalDetId id () const {return HcalDetId (mId);}
  /// charges corresponding to one timeslice of the channel
  float charge (unsigned i) const {return (i < FRAMESIZE) ? mCharge[i] : -1.;}
  /// array of 10 charges corresponding to one channel
  const float* getFrame () const {return mCharge;}
 private:
  uint32_t mId;
  float mCharge [10];
};

/// printout
std::ostream& operator<< (std::ostream&, const HPDNoiseDataFrame&);
    
#endif
