#ifndef ECAL_FENIX_TCP_FORMAT_H
#define ECAL_FENIX_TCP_FORMAT_H

#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h"
#include <vector>

class EcalTPParameters;

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;


  /** 
   \class EcalFenixStripFormat
   \brief Formatting for Fenix Tcp
   *  input 10 bits from Ettot 
   *         1 bit from fgvb
   *         3 bits TriggerTowerFlag 
   *  output: 16 bits
   *  simple formatting
   *  
   */
class EcalFenixTcpFormat  {

 public:
  EcalFenixTcpFormat(const EcalTPParameters * ecaltpp, bool tccFormat, bool debug, bool famos, int binOfMax); 
  virtual ~EcalFenixTcpFormat();
  virtual std::vector<int> process(std::vector<int>,std::vector<int>) {  std::vector<int> v;return v;}
  //  void process(std::vector<int> &Et, std::vector<int> &fgvb, int eTTotShift, std::vector<EcalTriggerPrimitiveSample> & out) ;
 void process(std::vector<int> &Et, std::vector<int> &fgvb,  int eTTotShift, std::vector<EcalTriggerPrimitiveSample> & out, std::vector<EcalTriggerPrimitiveSample> & outTcc) ;
  void setParameters(int SM, int towerInSM)  ;

 private:
  const EcalTPParameters *ecaltpp_ ;
  std::vector<unsigned int> const * lut_ ;
  bool tcpFormat_;
  bool debug_;
  bool famos_;
  unsigned int binOfMax_;
};

#endif
