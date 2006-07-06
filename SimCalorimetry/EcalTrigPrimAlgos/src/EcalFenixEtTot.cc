using namespace std;
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtTot.h>
#include <iostream>

namespace tpg {

// global type definitions for class implementation in source file defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_source> <tag_value>;

  EcalFenixEtTot::EcalFenixEtTot(){
  }
  EcalFenixEtTot::~EcalFenixEtTot(){
  }

  vector<int> EcalFenixEtTot::process(const vector<EBDataFrame> &calodatafr){
    std::vector<int> out;
    return out;
  }

  vector<int> EcalFenixEtTot::process(vector<vector <int> >  bypasslinout){
    std::vector<int> output(SIZEMAX);
    for (int i =0;i<SIZEMAX;i++){
     output[i]=0;
    }
    //    std::cout<<" size of EtTot input is: "<<bypasslinout.size()<<endl;;
    for(unsigned int istrip=0;istrip<bypasslinout.size();istrip++){
      vector<int> temp= bypasslinout[istrip];
      for (unsigned int i=0;i<temp.size();i++) {
	//	std::cout<<" "<<temp[i];
	output[i]+=temp[i];
	if(output[i]>0X3FFFF)output[i]=0X3FFFF;
      }
      //      std::cout<<endl;
    }
    return output;
    
  }
} /* End of namespace tpg */

