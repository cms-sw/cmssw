#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtTot.h>
#include <iostream>

  EcalFenixEtTot::EcalFenixEtTot(){
  }
  EcalFenixEtTot::~EcalFenixEtTot(){
  }

  std::vector<int> EcalFenixEtTot::process(const std::vector<EBDataFrame *> &calodatafr){
    std::vector<int> out;
    return out;
  }

  std::vector<int> EcalFenixEtTot::process(std::vector<std::vector <int> >  bypasslinout){
    std::vector<int> output(SIZEMAX);
    for (int i =0;i<SIZEMAX;i++){
     output[i]=0;
    }
    //    std::cout<<" size of EtTot input is: "<<bypasslinout.size()<<endl;;
    for(unsigned int istrip=0;istrip<bypasslinout.size();istrip++){
      std::vector<int> temp= bypasslinout[istrip];
      for (unsigned int i=0;i<temp.size();i++) {
	//	std::cout<<" "<<temp[i];
	output[i]+=temp[i];
	if(output[i]>0X3FFFF)output[i]=0X3FFFF;
      }
      //      std::cout<<endl;
    }
    return output;
    
  }

