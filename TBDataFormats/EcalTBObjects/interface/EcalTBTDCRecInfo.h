#ifndef RECECAL_ECALTBTDCRECINFO_H
#define RECECAL_ECALTBTDCRECINFO_H 1

#include <ostream>

/** \class EcalTBTDCRecInfo
 *  Simple container for TDC reconstructed informations 
 *
 *
 *  $Id: $
 */


class EcalTBTDCRecInfo {
 public:

  EcalTBTDCRecInfo() {};
  EcalTBTDCRecInfo(const float& offset): offset_(offset)
    {
    };
  
  ~EcalTBTDCRecInfo() {};
  
  float offset() const { return offset_; }

 private:

  float offset_;
  
};

std::ostream& operator<<(std::ostream&, const EcalTBTDCRecInfo&);
  
#endif
