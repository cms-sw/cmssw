#ifndef SimDataFormats_TrackerDigiSimLink_StripDigiSimLink_h
#define SimDataFormats_TrackerDigiSimLink_StripDigiSimLink_h

#include "boost/cstdint.hpp"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

class StripDigiSimLink {
 public:
  StripDigiSimLink(unsigned int ch, unsigned int tkId, unsigned int counter, EncodedEventId e,float a ):
    chan(ch),simTkId(tkId), CFpos(counter), eId(e) , fract(a) {;}

  StripDigiSimLink(unsigned int ch, unsigned int tkId, EncodedEventId e,float a ):
    chan(ch),simTkId(tkId), CFpos(0), eId(e) , fract(a) {;}

    StripDigiSimLink():chan(0),simTkId(0),CFpos(0),eId(0), fract(0) {;}

  ~StripDigiSimLink(){;}

  unsigned int   channel()     const {return chan;}
  unsigned int   SimTrackId()  const {return simTkId;}
  unsigned int   CFposition()  const {return CFpos;}
  EncodedEventId eventId()     const {return eId;}
  float          fraction()    const {return fract;}

  inline bool operator< ( const StripDigiSimLink& other ) const { return channel() < other.channel(); }

 private:
  unsigned int chan;
  unsigned int simTkId;
  unsigned int CFpos; //position of the PSimHit in the CrossingFrame vector
  EncodedEventId eId;
  float    fract;
};
#endif


  
