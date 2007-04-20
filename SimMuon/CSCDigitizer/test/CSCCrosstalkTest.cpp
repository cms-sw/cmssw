#include "SimMuon/CSCDigitizer/src/CSCStripAmpResponse.h"
#include "SimMuon/CSCDigitizer/src/CSCCrosstalkGenerator.h"
 
int main()
{
  CSCStripAmpResponse ampResponse(100, CSCStripAmpResponse::RADICAL);
  vector<float> binValues(300, 0.);
  CSCAnalogSignal signal(0, 1., binValues);

  float maxSlope = 0.;
  // sample every ns
  for(int i = 1; i < 300; ++i)
  {
    binValues[i] = ampResponse.calculateAmpResponse(i);
    float slope = binValues[i] - binValues[i-1];
    if(slope > maxSlope) maxSlope = slope;
  }

  CSCAnalogSignal signal(0, 1., binValues);

  float stripLength = 320.;

  CSCCrosstalkGenerator crosstalkGenerator(1/70., 0.03);
  CSCAnalogSignal crosstalkSignal = crosstalkGenerator.getCrosstalk(signal); 

  float maxFirst = 0., maxSecond = 0.;
  for(int i = 1; i < 300; ++i)
  {
    float v1 = signal.getBinValue(i);
    float v2 = crosstalkSignal.getBinValue(i);

    if(v1 > maxFirst) maxFirst = v1;
    if(v2 > maxSecond) maxSecond = v2;

     std::cout << "RAIOT " << v2/v1 << std::endl;
  }

   std::cout << "CROSSTALK " << maxSecond / maxFirst << std::endl;
}

