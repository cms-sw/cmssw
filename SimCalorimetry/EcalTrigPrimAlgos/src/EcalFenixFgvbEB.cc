#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixFgvbEB.h>


  std::vector<int> EcalFenixFgvbEB::process( std::vector<int> add_out, std::vector<int> maxof2_out) {

    std::vector<int> output(add_out.size());
    for (unsigned int i =0;i<add_out.size();i++) {
      if (add_out[i]>0 && float(maxof2_out[i])/float(add_out[i]) > .9 ) output[i]=1;
    }
    return output;
  }


