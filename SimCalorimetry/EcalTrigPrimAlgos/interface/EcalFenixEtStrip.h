#ifndef ECAL_FENIX_ET_STRIP_H
#define ECAL_FENIX_ET_STRIP_H
#include <vector>

#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixChip.h"

//#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVAdder.h>

//class EBDataFrame;

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;


  /** 
     \class EcalFenixEtStrip

     class for calculation of Et for Fenix strip
   *  input: 5x18 bits
   *  output: 18 bits representing sum
   *  
   *  sum method gets vector of CaloTimeSamples
   *  as input (steph comment : Ursula, why CaloTimeSample ?)
   *  simple sum, test for max?
   *  max in h4ana is 0x3FFFF 
   *  
   *  ---> if overflow sum= (2^18-1)
   */

//class EcalFenixEtStrip : public EcalVAdder {
class EcalFenixEtStrip  {
 private:

 public:
  EcalFenixEtStrip();
  virtual ~EcalFenixEtStrip();
  //  virtual std::vector<int> process(const std::vector<EBDataFrame *> &); 
  //    template <class T>  std::vector<int> process(const std::vector<T *> &); 
  std::vector<int> process(const std::vector<std::vector<int> > linout);
};

#endif
