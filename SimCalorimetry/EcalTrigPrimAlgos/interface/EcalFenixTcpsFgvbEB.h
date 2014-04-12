#ifndef ECAL_FENIXTCP_SFGVB_EB_H
#define ECAL_FENIXTCP_SFGVB_EB_H

#include <vector>
#include <stdint.h>

/** 
    \class EcalFenixTcpsFgvbEB
    \brief calculation of strip Fgvb for Fenix Tcp, format barrel
    *  calculates fgvb for the barrel
    *  
    * Takes the OR of all strip bits 
    */
    
class EcalFenixTcpsFgvbEB  {
 public:
   EcalFenixTcpsFgvbEB();
   virtual ~EcalFenixTcpsFgvbEB();

   void process( std::vector <std::vector<int> > &bypasslin_out,int nStr, int bitMask, std::vector<int> & output);
};
#endif
