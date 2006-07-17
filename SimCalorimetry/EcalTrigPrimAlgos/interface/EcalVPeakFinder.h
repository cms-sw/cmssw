#ifndef ECAL_V_PEAKFINDER_H
#define ECAL_V_PEAKFINDER_H
#include <stdio.h>
#include <vector>

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;

class EcalVPeakFinder {


public:
  virtual std::vector<int> process(std::vector<int>) = 0;
 
};

#endif
