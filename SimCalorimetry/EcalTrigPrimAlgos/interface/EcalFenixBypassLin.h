#ifndef ECAL_FENIX_BYPASS_LIN_H
#define ECAL_FENIX_BYPASS_LIN_H

//#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVLinearizer.h>
#include <vector>

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;


  /** 
  \class EcalFenixBypassLin
  \brief Linearisation for Tcp
   *  input: 16 bits
   *  output: 12 bits +1 going to fgvb (???)
   *  
   *      ----> c un output de 13 bits, le 13 eme bit est le resultat du fvgb du FenixStrip
   */
//class EcalFenixBypassLin : public EcalVLinearizer {
class EcalFenixBypassLin  {


 public:
  EcalFenixBypassLin();
  virtual ~EcalFenixBypassLin();

  //  virtual EBDataFrame process(EBDataFrame &); //FIXME: efficiency.. 
  //  std::vector<int> process(std::vector<int>);
 void process(std::vector<int>&, std::vector<int>& );
  };


#endif
