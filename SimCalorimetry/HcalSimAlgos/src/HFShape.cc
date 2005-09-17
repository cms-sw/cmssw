#include "SimCalorimetry/HcalSimAlgos/interface/HFShape.h"
#include <cmath>
  
namespace cms {
  
  void HFShape::computeShapeHF()
  {
  
    //  cout << endl << " ===== computeShapeHF  !!! " << endl << endl;
  
    const float ts = 3.0;           // time constant in   t * exp(-(t/ts)**2)
  
    // first create pulse shape over a range of time 0 ns to 255 ns in 1 ns steps
    nbin = 256;
    std::vector<float> ntmp(nbin,0.0);  // 
  
    int j;
    float norm;
  
    // HF SHAPE
    norm = 0.0;
    for( j = 0; j < 3 * ts && j < nbin; j++){
      ntmp[j] = ((float)j)*exp(-((float)(j*j))/(ts*ts));
      norm += ntmp[j];
    }
    // normalize pulse area to 1.0
    for( j = 0; j < 3 * ts && j < nbin; j++){
      ntmp[j] /= norm;
  
      //    cout << " nt [" << j << "] = " <<  ntmp[j] << endl;
  
    }
    nt = ntmp;
  }
  
  double HFShape::operator () (double time_) const
  {
  
    // return pulse amplitude for request time in ns
    int jtime;
    jtime = (int)(time_+0.5);
  
    if(jtime >= 0 && jtime < nbin)
      return nt[jtime];
    else 
      return 0.0;
  }
  
  double HFShape::derivative (double time_) const
  {
    return 0.0;
  }
  
}
  
