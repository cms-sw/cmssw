#ifndef GEMObjects_GEMStripNoise_h
#define GEMObjects_GEMStripNoise_h

#include<vector>

class GEMStripNoise 
{
 public:
  
  struct StripNoiseItem {
    int dpid;
    float stripNoise;
  };
  
  GEMStripNoise(){}
  ~GEMStripNoise(){}
  
  std::vector<StripNoiseItem> const & getStripNoiseVector() const {return stripNoiseVector_;}

 private:

  std::vector<StripNoiseItem> stripNoiseVector_; 
};

#endif  
