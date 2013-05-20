#ifndef GEMObjects_GEMStripTiming_h
#define GEMObjects_GEMStripTiming_h

#include<vector>

class GEMStripTiming 
{
 public:
  
  struct StripTimingItem {
    int dpid;
    float timeCalibrationOffset;
  };
  
  GEMStripTiming(){}
  ~GEMStripTiming(){}
  
  std::vector<StripTimingItem> const & getStripTimingVector() const {return stripTimingVector_;}

 private:

  std::vector<StripTimingItem> stripTimingVector_; 
};

#endif  
