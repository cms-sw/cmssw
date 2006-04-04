#ifndef TRACKINGOBJECTS_PIXELDIGISIMLINK_H
#define TRACKINGOBJECTS_PIXELDIGISIMLINK_H

//typedef std::pair<unsigned int ,unsigned int > PixelDigiSimLink;
class PixelDigiSimLink {
public:
  PixelDigiSimLink(unsigned int ch, unsigned int tkId, float a ){
    chan=ch;
    simTkId=tkId;
    fract=a;
  };
  PixelDigiSimLink(){
   chan=0;
   simTkId=0;
   fract=0;};
  ~PixelDigiSimLink(){};
  unsigned int channel() const{return chan;};
  unsigned int SimTrackId() const{return simTkId;};
  float fraction() const{return fract;};
 private:
  unsigned int chan;
  unsigned int simTkId;
  float fract;
  };
#endif 
