#include "SimMuon/CSCDigitizer/src/CSCStripAmpResponse.h"
#include "SimMuon/CSCDigitizer/src/CSCCrosstalkGenerator.h"
#include "SimMuon/CSCDigitizer/src/CSCAnalogSignal.h"
#include <iostream>
 
int main()
{
  CSCStripAmpResponse ampResponse(100, CSCStripAmpResponse::RADICAL);
  std::vector<float> binValues(200, 0.);

  float maxSlope = 0.;
  float timeBinSize = 25.;
  // sample every ns
  for(int i = 1; i < 200; ++i)
  {
    binValues[i] = ampResponse.calculateAmpResponse(i*timeBinSize);
    float slope = binValues[i] - binValues[i-1];
    if(slope > maxSlope) maxSlope = slope;
  }

  CSCAnalogSignal signal(0, timeBinSize, binValues);

  float stripLength = 320.;

  CSCCrosstalkGenerator crosstalkGenerator;
  crosstalkGenerator.setParameters(stripLength/70, 0., 0.02);
  CSCAnalogSignal crosstalkSignal = crosstalkGenerator.getCrosstalk(signal); 

std::cout << signal << std::endl;;
std::cout << crosstalkSignal << std::endl;
  float maxFirst = 0., maxSecond = 0.;
  for(int i = 1; i < 200; ++i)
  {
    float v1 = signal.getBinValue(i);
    float v2 = crosstalkSignal.getBinValue(i);

    if(v1 > maxFirst) maxFirst = v1;
    if(v2 > maxSecond) maxSecond = v2;

    // print every 10 ns
     std::cout << "RATIO " << i<< " " << v2/v1 << std::endl;
  }

   std::cout << "CROSSTALK " << maxSecond / maxFirst << std::endl;
std::cout << "MAXSLOPE " << maxSlope << std::endl;
}

