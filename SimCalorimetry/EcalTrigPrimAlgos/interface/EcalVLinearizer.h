#ifndef ECAL_V_LINEARIZER_H
#define ECAL_V_LINEARIZER_H
#include <stdio.h>

class EBDataFrame;
// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;

  /** 
     \class EcalVLinearizer

     \brief abstract base class for linearization
  */

class EcalVLinearizer {
  /* {src_lang=Cpp}*/



 public:
  virtual EBDataFrame  process(EBDataFrame &) =0; //FIXME: efficiency!
}; 


#endif
