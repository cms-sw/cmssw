#ifndef ECAL_V_AMPLITUDE_FILTER_H
#define ECAL_V_AMPLITUDE_FILTER_H
#include <stdio.h>
#include <vector>


// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;

class EcalVAmplitudeFilter {
  /* {src_lang=Cpp}*/
 public:
  virtual std::vector<int> process(std::vector<int>) =0;
};


#endif
