#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixMaxof2.h>


namespace tpg {

// global type definitions for class implementation in source file defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_source> <tag_value>;
  EcalFenixMaxof2::EcalFenixMaxof2(){
  }
  
  EcalFenixMaxof2::~EcalFenixMaxof2(){
  }
  
  std::vector<int> EcalFenixMaxof2::process(std::vector<std::vector <int> > bypasslinout){

    int nstrip= bypasslinout.size();
    
    std::vector<int> output(SIZEMAX);
    for (int i=0;i<SIZEMAX;i++){
      output[i]=0;
    }
    std::vector<std::vector<int> >  sumby2(nstrip-1);
    for (int i2strip =0;i2strip<nstrip-1;i2strip++){
      for (int i=0;i<SIZEMAX;i++){
	sumby2[i2strip].push_back(0);
      }
    }
    for (int i=0;i<SIZEMAX;i++){
      if (nstrip-1==0){
	output[i]=bypasslinout[0][i];
      }
      for ( int i2strip=0; i2strip< nstrip-1;i2strip++){ 
	sumby2[i2strip][i]= bypasslinout[i2strip][i]+bypasslinout[i2strip+1][i];
	if (sumby2[i2strip][i]>output[i]){
	  output[i]=sumby2[i2strip][i];
	}
      }
    }
    return output;
  }

} /* End of namespace tpg */

