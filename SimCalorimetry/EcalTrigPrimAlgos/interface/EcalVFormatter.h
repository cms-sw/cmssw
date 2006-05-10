using namespace std;
#ifndef ECAL_VFORMATTER_H
#define ECAL_VFORMATTER_H

#include <stdio.h>
#include <vector>

namespace tpg {

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;

  /** 
     \class EcalVFormatter

     \brief abstract base class for formatters in fenix chips
  */

class EcalVFormatter {
  /* {src_lang=Cpp}*/



 public:
  virtual vector<int> process(vector<int>,vector<int>) = 0;
 };

} /* End of namespace tpg */

#endif
