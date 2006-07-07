#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFormat.h>


namespace tpg {

  EcalFenixTcpFormat::EcalFenixTcpFormat() { 

  }
  EcalFenixTcpFormat::~EcalFenixTcpFormat() {
  }

 
  void EcalFenixTcpFormat::process(std::vector<int> &Et, std::vector<int> &fgvb, std::vector<EcalTriggerPrimitiveSample> & out){
    
    //FIXME: must be configurable
    //    const double towerLowThreshold = 2.5; //in GeV
    //    const double towerHighThreshold = 5.; //in GeV
    const int towerLowThreshold = 18;  //in adc values FIXME: we add a factor of 4 due to shift by 2 in FenixStrip
    const int towerHighThreshold = 36; //in adc values

   //FIXME: should not be hardcoded!!
    //FIXME: value valid only for barrel! (GeV/ADC~0.06 for EE)
    //    double adc2GeV = 0.035;
    //double adc2GeV = 0.035*4.; 
    
    for (unsigned int i=0; i<Et.size();++i) {
      if (Et[i]>0xFF) Et[i]=0xFF;
      //computes trigger tower flag:
      //FIXME: check if < or <=.
      int ttFlag;

      //FIXME: flag code should go to some enum 
      if(Et[i]<towerLowThreshold){ //Low interest
	ttFlag = 0;
      } else if(Et[i] < towerHighThreshold){ //Mid interest
	ttFlag = 0x1;
      } else{ //etGeV>=towerHighThreshold => High interest
	ttFlag = 0x3;
      }
      
      out.push_back(EcalTriggerPrimitiveSample( Et[i],fgvb[i],ttFlag)); 
    }
		    
  }
} /* End of namespace tpg */

