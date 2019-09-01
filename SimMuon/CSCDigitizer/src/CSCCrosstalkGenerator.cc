#include "SimMuon/CSCDigitizer/src/CSCAnalogSignal.h"
#include "SimMuon/CSCDigitizer/src/CSCCrosstalkGenerator.h"

CSCAnalogSignal CSCCrosstalkGenerator::getCrosstalk(const CSCAnalogSignal &inputSignal) const {
  int nBins = inputSignal.getSize();
  float binSize = inputSignal.getBinSize();
  std::vector<float> binValues(nBins);

  for (int outputBin = 0; outputBin < nBins; ++outputBin) {
    float aTime = outputBin * binSize - theDelay;
    float slope = inputSignal.getValue(aTime) - inputSignal.getValue(aTime - 1.);
    binValues[outputBin] = slope * theCrosstalk + theResistiveFraction * inputSignal.getValue(aTime);
  }

  return CSCAnalogSignal(0, binSize, binValues, 0., inputSignal.getTimeOffset());
}

float CSCCrosstalkGenerator::ratio(const CSCAnalogSignal &crosstalkSignal, const CSCAnalogSignal &signal) const {
  float maxFirst = 0., maxSecond = 0.;
  int nbins = signal.getSize();
  for (int i = 1; i < nbins; ++i) {
    float v1 = signal.getBinValue(i);
    float v2 = crosstalkSignal.getBinValue(i);

    if (v1 > maxFirst)
      maxFirst = v1;
    if (v2 > maxSecond)
      maxSecond = v2;
  }

  return maxSecond / maxFirst;
}
