#include "SimMuon/CSCDigitizer/src/CSCCrosstalkGenerator.h"
#include "SimMuon/CSCDigitizer/src/CSCAnalogSignal.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

CSCAnalogSignal CSCCrosstalkGenerator::getCrosstalk(const CSCAnalogSignal & inputSignal) const {
  int nBins = inputSignal.getSize();
  float binSize = inputSignal.getBinSize();
  std::vector<float> binValues(nBins);
  
  for(int outputBin = 0; outputBin < nBins; ++outputBin) {
    float aTime = outputBin*binSize - theDelay;
    float slope = inputSignal.getValue(aTime) - inputSignal.getValue(aTime-1.);
    binValues[outputBin] = slope * theCrosstalk 
                         + theResistiveFraction * inputSignal.getValue(aTime);
  }

  CSCAnalogSignal signal(0, binSize, binValues);
  return signal;
}
  
