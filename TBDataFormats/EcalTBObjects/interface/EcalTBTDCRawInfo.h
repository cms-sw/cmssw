#ifndef ECALTBTDCRAWINFO_H
#define ECALTBTDCRAWINFO_H 1

#include <ostream>
#include <vector>
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCSample.h"

/** \class EcalTBTDCRawInfo
      
$Id: EcalTBTDCRawInfo.h,v 1.1 2006/03/28 13:52:06 meridian Exp $
*/

class EcalTBTDCRawInfo {
 public:
  EcalTBTDCRawInfo() : data_(0) {};

  EcalTBTDCRawInfo(unsigned int channels) 
    {
      data_.reserve(channels);
    }; 

  /// Get methods
  unsigned int size() const { return data_.size(); }
  unsigned int channels() const { return data_.size(); }
  const EcalTBTDCSample& operator[](unsigned int i) const { return data_[i]; }
  const EcalTBTDCSample& sample(unsigned int i) const { return data_[i]; }
    
  /// Set methods
  void setSize(unsigned int size)
    {
      // data_.reserve(size);
      data_.resize(size);
    };
  
  void setSample(unsigned int i, const EcalTBTDCSample& sam) 
    {
      if (data_.size() < i )
	// data_.reserve(i);
	data_.resize(i);
      data_[i]=sam; 
    };
  
 private:

  std::vector<EcalTBTDCSample> data_;
};


std::ostream& operator<<(std::ostream& s, const EcalTBTDCRawInfo& digi);



#endif
