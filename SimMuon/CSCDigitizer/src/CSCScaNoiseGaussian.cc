#include "SimMuon/CSCDigitizer/src/CSCScaNoiseGaussian.h"
#include "CLHEP/Random/RandGaussQ.h"

CSCScaNoiseGaussian::CSCScaNoiseGaussian(double analogNoise, double pedestal, double pedestalWidth)
: analogNoise_(analogNoise),
  pedestal_(pedestal),
  pedestalWidth_(pedestalWidth)
{
}

void CSCScaNoiseGaussian::noisify(const CSCDetId & layer, 
                                 CSCAnalogSignal & signal)
{
  fill(layer, signal.getElement());
  for(int i = 0; i < signal.getSize(); ++i) {
    // since this is just the analog signal, no pedestals yet.
    // it's in ADC counts, so someone will have to convert to
    // fC or mV later
    signal[i] += static_cast<int>(RandGaussQ::shoot(0., analogNoise_));
  }
}


void CSCScaNoiseGaussian::addPedestal(const CSCDetId & layer, 
                                      CSCStripDigi & digi)
{
  fill(layer, digi.getStrip());
  std::vector<int> result = digi.getADCCounts();
  for(unsigned i = 0; i < result.size(); ++i) {
    result[i] += static_cast<int>(RandGaussQ::shoot(pedestal_, pedestalWidth_));
  }
  digi.setADCCounts(result);
}

