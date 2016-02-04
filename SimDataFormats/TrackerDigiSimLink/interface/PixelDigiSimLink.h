#ifndef TRACKINGOBJECTS_PIXELDIGISIMLINK_H
#define TRACKINGOBJECTS_PIXELDIGISIMLINK_H

#include "boost/cstdint.hpp"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

//typedef std::pair<unsigned int ,unsigned int > PixelDigiSimLink;
class PixelDigiSimLink {
public:
  PixelDigiSimLink(unsigned int ch, unsigned int tkId, EncodedEventId e, float a ){
    chan=ch;
    simTkId=tkId;
    fract=a;
    eId=e;
  };
  PixelDigiSimLink():eId(0){
    chan=0;
    simTkId=0;
    fract=0;
  };
  ~PixelDigiSimLink(){};
  unsigned int channel() const{return chan;};
  unsigned int SimTrackId() const{return simTkId;};
  EncodedEventId eventId() const{return eId;}
  float fraction() const{return fract;};


  inline bool operator< ( const PixelDigiSimLink& other ) const { return fraction() < other.fraction(); }

 private:
  unsigned int chan;
  unsigned int simTkId;
  EncodedEventId eId;
  float fract;
  };
#endif 
