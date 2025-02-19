#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpsFgvbEB.h>

//---------------------------------------------------------------
EcalFenixTcpsFgvbEB::EcalFenixTcpsFgvbEB()
{
}//---------------------------------------------------------------
EcalFenixTcpsFgvbEB::~EcalFenixTcpsFgvbEB()
{
}
//---------------------------------------------------------------
void EcalFenixTcpsFgvbEB::process(std::vector<std::vector<int> > & bypasslin_out, int nStr,int bitMask,std::vector<int> & output)
{
  //  std::vector<int> indexLut(output.size());
  
  for (unsigned int i=0;i<output.size();i++) {
    output[i]=0;
  }
    
  for (unsigned int i=0;i<output.size();i++) {
    int towRes = 0;
    for (int istrip=0;istrip<nStr;istrip++) {
      int res = (bypasslin_out[istrip])[i]; 
      res = (res >>bitMask) & 1; //res is sFGVB at this stage
      towRes |= res;
    }
     
    output[i]= towRes;
  }
  return;
} 

