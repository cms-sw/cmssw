#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFormat.h>


namespace tpg {

  EcalFenixTcpFormat::EcalFenixTcpFormat() { 

  }
  EcalFenixTcpFormat::~EcalFenixTcpFormat() {
  }

  std::vector<int> EcalFenixTcpFormat::process(std::vector<int> Et, std::vector<int> fgvb){
    vector<int> out (Et.size());

    int ttFlag=0;  //to be fixed, taken from Lookup table
    for (unsigned int i=0; i<Et.size();++i) {
      if (Et[i]>0xFF) Et[i]=0xFF;
      out[i]=((ttFlag&0x3)<<9)|(Et[i]&0xFF);
      out[i]=out[i] | fgvb[i]<<8;
    }
    return out;
  }

} /* End of namespace tpg */

