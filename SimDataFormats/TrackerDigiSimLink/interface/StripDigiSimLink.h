#ifndef TRACKINGOBJECTS_STRIPDIGISIMLINK_H
#define TRACKINGOBJECTS_STRIPDIGISIMLINK_H

#include "boost/cstdint.hpp"

class StripDigiSimLink {
 public:
  StripDigiSimLink(const uint16_t& ch, const uint16_t& tkId, const float& a ):chan(ch),simTkId(tkId),fract(a){;}
  
  StripDigiSimLink():chan(0),simTkId(0),fract(0){;}

  ~StripDigiSimLink(){;}

  inline const uint16_t&  channel()     const {return chan;}
  inline const uint16_t&  SimTrackId()  const {return simTkId;}
  inline const float&   fraction()    const {return fract;}

  inline bool operator< ( const StripDigiSimLink& other ) const { return channel() < other.channel(); }

 private:
  uint16_t chan;
  uint16_t simTkId;
  float    fract;
};
#endif 
  
