#include "SimMuon/CSCDigitizer/src/CSCStripConditions.h"
#include "CLHEP/Random/RandGaussQ.h"

float CSCStripConditions::smearedGain(const CSCDetId & detId, int channel) const
{
  return RandGaussQ::shoot( gain(detId, channel), gainVariance(detId, channel) );
}


void CSCStripConditions::noisify(const CSCDetId & detId, CSCAnalogSignal & signal)
{
  const int nScaBins = 8;
  const float scaBinSize = 50.;
  std::vector<float> binValues(nScaBins, 0.);
  // use a temporary signal, in case we have to rebin
  CSCAnalogSignal tmpSignal(signal.getElement(), scaBinSize, binValues,
                            0., signal.getTimeOffset());

  fetchNoisifier(detId, signal.getElement() );
  theNoisifier->noisify(tmpSignal);

  signal.superimpose(tmpSignal);
}

 
float CSCStripConditions::analogNoise(const CSCDetId & detId, int channel) const
{
  return sqrt(2) * pedestalVariance(detId, channel) / gain(detId, channel);
}


