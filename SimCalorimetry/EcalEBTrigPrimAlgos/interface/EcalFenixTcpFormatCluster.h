#ifndef ECAL_FENIX_TCP_FORMAT_CLU_H
#define ECAL_FENIX_TCP_FORMAT_CLU_H

#include "DataFormats/EcalDigi/interface/EcalEBClusterTriggerPrimitiveSample.h"
#include <vector>

class EcalTPGLutGroup ;
class EcalTPGLutIdMap;
class EcalTPGTowerStatus;
class EcalTPGSpike;

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
class EcalFenixTcpFormatCluster  {

 public:
  EcalFenixTcpFormatCluster(bool tccFormat, bool debug, bool famos, int binOfMax); 
  virtual ~EcalFenixTcpFormatCluster();

  void process(std::vector<int>&,std::vector<int>&);
  void process(std::vector<int> &Et, std::vector<int> &fgvb, std::vector<int> &sfgvb, int eTTotShift, std::vector<EcalEBClusterTriggerPrimitiveSample> & out, std::vector<EcalEBClusterTriggerPrimitiveSample> & outTcc, bool isInInnerRings) ;
  void setParameters(uint32_t towid,const EcalTPGLutGroup *ecaltpgLutGroup,const EcalTPGLutIdMap *ecaltpgLut, const EcalTPGTowerStatus *ecaltpgbadTT, const EcalTPGSpike * ecaltpgSpike);

 private:
  const unsigned int * lut_ ;
  const uint16_t * badTTStatus_;
  uint16_t status_;
  bool tcpFormat_;
  bool debug_;
  bool famos_;
  unsigned int binOfMax_;
  uint16_t spikeZeroThresh_;
};

#endif
