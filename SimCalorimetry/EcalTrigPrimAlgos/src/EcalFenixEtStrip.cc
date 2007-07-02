#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtStrip.h>
  
  EcalFenixEtStrip::EcalFenixEtStrip(){
  }
  EcalFenixEtStrip::~EcalFenixEtStrip(){
  }
  
  
  std::vector<int> EcalFenixEtStrip::process(const std::vector<std::vector<int> > linout)
  {
    std::vector<int> output(SIZEMAX);
    for (int i =0;i<SIZEMAX;i++){
     output[i]=0;
    }
    for(unsigned int ixtal=0;ixtal<linout.size();ixtal++){
      for (int i=0;i<SIZEMAX;i++) {
  	output[i]+=(((linout[ixtal]))[i]);
     }
    }
    for (int i=0;i<SIZEMAX;i++) {
      output[i]>>2 ;
      if(output[i]>0X3FFFF)output[i]=0X3FFFF;
    }
    return output;
  }


