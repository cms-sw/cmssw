#ifndef L1EGammaCrystalsCluster_h
#define L1EGammaCrystalsCluster_h

#include <vector>

namespace l1slhc
{

  class L1EGCrystalCluster {
  public:
    float et ;
    float eta ;
    float phi ;
    float e ;
    float x ;
    float y ;
    float z ;
    float hovere ;

    float ECALiso ;
    float ECALetPUcorr;
    
  };
  
  
  // Concrete collection of output objects (with extra tuning information)
  typedef std::vector<l1slhc::L1EGCrystalCluster> L1EGCrystalClusterCollection;
}
#endif

