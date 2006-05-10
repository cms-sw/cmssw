using namespace std; // necessary!
#include <iostream>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtStrip.h>
#include <DataFormats/EcalDigi/interface/EBDataFrame.h>


namespace tpg {
  
  EcalFenixEtStrip::EcalFenixEtStrip(){
  }
  EcalFenixEtStrip::~EcalFenixEtStrip(){
  }
  
  
  
  vector<int> EcalFenixEtStrip::process(vector<EBDataFrame> linout)
  {
    std::vector<int> output(SIZEMAX);
    for (int i =0;i<SIZEMAX;i++){
     output[i]=0;
    }
    //    cout<<" size of EtStrip input is: "<<linout.size()<<endl;;
    for(unsigned int ixtal=0;ixtal<linout.size();ixtal++){
      for (int i=0;i<SIZEMAX;i++) {
	EBDataFrame temp= linout[ixtal];
	//	cout<<"value is: "<<temp[i]<<endl;
 	output[i]+=(linout[ixtal])[i].adc();
	if(output[i]>0X3FFFF)output[i]=0X3FFFF;
      }
    }
    return output;
  }

// global type definitions for class implementation in source file defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_source> <tag_value>;

} /* End of namespace tpg */

