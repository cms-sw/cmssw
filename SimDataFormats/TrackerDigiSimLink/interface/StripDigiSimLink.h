#ifndef TRACKINGOBJECTS_STRIPDIGISIMLINK_H
#define TRACKINGOBJECTS_STRIPDIGISIMLINK_H

#include "boost/cstdint.hpp"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

class StripDigiSimLink {
 public:
  StripDigiSimLink(unsigned int ch, unsigned int tkId, EncodedEventId e,float a ):
    chan(ch),simTkId(tkId),fract(a),eId(e){;}

  StripDigiSimLink():chan(0),simTkId(0),fract(0),eId(0){;}

  ~StripDigiSimLink(){;}

  unsigned int  channel()     const {return chan;}
  unsigned int  SimTrackId()  const {return simTkId;}
  EncodedEventId  eventId()  const {return eId;}
  float   fraction()    const {return fract;}

  inline bool operator< ( const StripDigiSimLink& other ) const { return channel() < other.channel(); }

 private:
  unsigned int chan;
  unsigned int simTkId;
  float    fract;
  EncodedEventId eId;
};
#endif


  
