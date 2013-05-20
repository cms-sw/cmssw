#ifndef GEMObjects_GEMStripEfficiency_h
#define GEMObjects_GEMStripEfficiency_h

#include<vector>

class GEMStripEfficiency 
{
 public:
  
  struct StripEfficiencyItem {
    int dpid;
    float stripEfficiency;
  };
  
  GEMStripEfficiency(){}
  ~GEMStripEfficiency(){}
  
  std::vector<StripEfficiencyItem> const & getStripEfficiencyVector() const {return stripEfficiencyVector_;}

 private:

  std::vector<StripEfficiencyItem> stripEfficiencyVector_; 
};

#endif  
