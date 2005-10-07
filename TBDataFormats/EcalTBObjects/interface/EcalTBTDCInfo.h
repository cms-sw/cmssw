#ifndef ECALTBTDCINFO_H
#define ECALTBTDCINFO_H 1

#include <ostream>
#include <vector>
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCSample.h"



/** \class EcalTBTDCInfo
      
$Id: $
*/

class EcalTBTDCInfo {
 public:
  EcalTBTDCInfo(); 
    
  int size() const { return size_; }
    
  const EcalTBTDCSample& operator[](int i) const { return data_[i]; }
  const EcalTBTDCSample& sample(int i) const { return data_[i]; }
    
  void setSize(int size);
  void setSample(int i, const EcalTBTDCSample& sam) { data_[i]=sam; }
    
  static const int MAXSAMPLES = 255; //just to initialize the vector
 private:

  int size_;
  std::vector<EcalTBTDCSample> data_;
};


std::ostream& operator<<(std::ostream& s, const EcalTBTDCInfo& digi);



#endif
