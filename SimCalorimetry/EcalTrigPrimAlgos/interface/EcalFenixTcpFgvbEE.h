#ifndef ECAL_FENIXTCP_FGVB_EE_H
#define ECAL_FENIXTCP_FGVB_EE_H

//#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVFgvb.h>
#include <vector>

class EcalTPParameters;

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;


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
   const EcalTPParameters * ecaltpp_ ;
   std::vector<unsigned int> const * params_ ;
    
 public:
   EcalFenixTcpFgvbEE(const EcalTPParameters * ecaltpp);
   virtual ~EcalFenixTcpFgvbEE();
   void setParameters(int SM, int towNum);

   void process( std::vector <std::vector<int> > &bypasslin_out,int nStr, int bitMask, std::vector<int> & output);
};


#endif
