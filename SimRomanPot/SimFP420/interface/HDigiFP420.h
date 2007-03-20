#ifndef HDigiFP420_h
#define HDigiFP420_h

class HDigiFP420 {
public:

  //typedef unsigned int ChannelType;

  HDigiFP420() : strip_(0), adc_(0) {}

  HDigiFP420( int strip, int adc) : strip_(strip), adc_(adc) {}
    HDigiFP420( short strip, short adc) : strip_(strip), adc_(adc) {}

  // Access to digi information
  int strip() const   {return strip_;}
  int adc() const     {return adc_;}
  int channel() const {return strip();}

private:
  short strip_;
  short adc_;
};

// Comparison operators
inline bool operator<( const HDigiFP420& one, const HDigiFP420& other) {
  return one.channel() < other.channel();
}

#endif
