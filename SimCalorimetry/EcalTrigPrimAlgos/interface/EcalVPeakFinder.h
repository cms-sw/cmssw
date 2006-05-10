using namespace std;
#ifndef ECAL_V_PEAKFINDER_H
#define ECAL_V_PEAKFINDER_H
#include <stdio.h>
#include <vector>

namespace tpg {

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;

class EcalVPeakFinder {


public:
  virtual vector<int> process(vector<int>) = 0;
 
};

} /* End of namespace tpg */

#endif
