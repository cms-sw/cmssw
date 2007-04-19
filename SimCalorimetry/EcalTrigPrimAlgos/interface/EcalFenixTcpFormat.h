#ifndef ECAL_FENIX_TCP_FORMAT_H
#define ECAL_FENIX_TCP_FORMAT_H

#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVFormatter.h>
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h"
#include <vector>

class EcalTPParameters;


// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;


  /** 
   \class EcalFenixStripFormat
   \brief Formatting for Fenix strip
   *  input 10 bits from Ettot 
   *         1 bit from fgvb
   *         3 bits TriggerTowerFlag 
   *  output: 16 bits
   *  simple formatting
   *  
   */
class EcalFenixTcpFormat : public EcalVFormatter {


 public:
  EcalFenixTcpFormat(const EcalTPParameters * ecaltpp, bool tccFormat, bool debug);
  virtual ~EcalFenixTcpFormat();
  virtual std::vector<int> process(std::vector<int>,std::vector<int>) {  std::vector<int> v;return v;}
  void process(std::vector<int> &Et, std::vector<int> &fgvb, std::vector<EcalTriggerPrimitiveSample> & out, std::vector<EcalTriggerPrimitiveSample> & outTcc) ;
  void setParameters(int SM, int towerInSM)  ;

 private:
  const EcalTPParameters *ecaltpp_ ;
  std::vector<unsigned int> lut_ ;
  bool tcpFormat_;
  bool debug_;

};

#endif
