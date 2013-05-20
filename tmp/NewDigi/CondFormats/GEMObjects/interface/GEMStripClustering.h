#ifndef GEMObjects_GEMStripClustering_h
#define GEMObjects_GEMStripClustering_h

#include<vector>

class GEMStripClustering 
{
 public:
  
  struct StripClusteringItem {
    int dpid;
    float clusterSize;
  };
  
  GEMStripClustering(){}
  ~GEMStripClustering(){}
  
  std::vector<StripClusteringItem> const & getClusterSizeVector() const {return clusterSizeVector_;}

 private:

  std::vector<StripClusteringItem> clusterSizeVector_; 
};

#endif  
