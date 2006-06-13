#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFormat.h>


namespace tpg {

  EcalFenixTcpFormat::EcalFenixTcpFormat() { 

  }
  EcalFenixTcpFormat::~EcalFenixTcpFormat() {
  }

  std::vector<int> EcalFenixTcpFormat::process(std::vector<int> Et, std::vector<int> fgvb){
    vector<int> out (Et.size());
    
    //FIXME: must configurabled
    const double towerLowThreshold = 2.5; //in GeV
    const double towerHighThreshold = 5.; //in GeV

    //FIXME: should not be hardcoded!!
    //FIXME: value valid only for barrel! (GeV/ADC~0.06 for EE)
    double adc2GeV = 0.035;
    
    for (unsigned int i=0; i<Et.size();++i) {
      if (Et[i]>0xFF) Et[i]=0xFF;
      //computes trigger tower flag:
      //FIXME: check if < or <=.
      int ttFlag;
      double etGeV=Et[i]*adc2GeV;
      //FIXME: flag code should go to some enum ?
      if(etGeV<towerLowThreshold){ //Low interest
	ttFlag = 0;
      } else if(etGeV < towerHighThreshold){ //Mid interest
	ttFlag = 0x1;
      } else{ //etGeV>=towerHighThreshold => High interest
	ttFlag = 0x3;
      }
      
      out[i]=((ttFlag&0x7)<<9)|(Et[i]&0xFF);
      out[i]=out[i] | fgvb[i]<<8;
    }
    return out;
  }
} /* End of namespace tpg */

