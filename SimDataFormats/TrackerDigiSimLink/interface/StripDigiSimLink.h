#ifndef TRACKINGOBJECTS_STRIPDIGISIMLINK_H
#define TRACKINGOBJECTS_STRIPDIGISIMLINK_H

//typedef std::pair<unsigned int ,unsigned int > StripDigiSimLink;
class StripDigiSimLink {
 public:
  StripDigiSimLink(unsigned int ch, unsigned int tkId, float a ){
    chan=ch;
    simTkId=tkId;
    fract=a;
  };
  
  StripDigiSimLink(){
    chan=0;
    simTkId=0;
    fract=0;
  };
  
  ~StripDigiSimLink(){};

  unsigned int channel(){return chan;};
  unsigned int SimTrackId(){return simTkId;};
  float fraction(){return fract;};
 private:
  unsigned int chan;
  unsigned int simTkId;
  float fract;
};
#endif 
  
