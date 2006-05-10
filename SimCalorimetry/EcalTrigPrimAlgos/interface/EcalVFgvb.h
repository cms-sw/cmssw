#ifndef ECAL_V_FGVB_H
#define ECAL_V_FGVB_H
#include <stdio.h>

namespace tpg {

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;

  /** 
     \class EcalVFgvb

     \brief abstract base class for calculation of Fgvb
  */
class EcalVFgvb {
  /* {src_lang=Cpp}*/



 public:
  virtual int process() =0;
  };

} /* End of namespace tpg */

#endif
