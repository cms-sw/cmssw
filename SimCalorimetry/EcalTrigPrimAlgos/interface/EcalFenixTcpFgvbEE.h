#ifndef ECAL_FENIXTCP_FGVB_EE_H
#define ECAL_FENIXTCP_FGVB_EE_H

#include <vector>
#include <stdint.h>

class EcalTPGFineGrainTowerEE;

/** 
    \class EcalFenixTcpFgvbEE
    \brief calculation of Fgvb for Fenix Tcp, format endcap
    *  calculates fgvb for the endcap
    *  
    *  
    *  input :  5x 11th  bit of Bypasslin ouput 
    *  output: 1 bit 
    *  
    *  
    *  the five bit_strips + 3 bits (nb of strips) (are used to make a LUT_tower address (max size :2**8=256)
    *  the output is 1 value.
    */
    
class EcalFenixTcpFgvbEE  {

 private:
   uint32_t fgee_lut_;
   std::vector<int> indexLut_;

 public:
   EcalFenixTcpFgvbEE(int maxNrSamples);
   virtual ~EcalFenixTcpFgvbEE();
   void setParameters(uint32_t towid, const EcalTPGFineGrainTowerEE *ecaltpgFineGrainTowerEE);

   void process( std::vector <std::vector<int> > &bypasslin_out,int nStr, int bitMask, std::vector<int> & output);
};
#endif
