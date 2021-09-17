#include "SimTracker/SiStripDigitizer/interface/SiGaussianTailNoiseAdder.h"
#include "CLHEP/Random/RandGaussQ.h"

SiGaussianTailNoiseAdder::SiGaussianTailNoiseAdder(float th)
    : threshold(th), genNoise(new GaussianTailNoiseGenerator()) {}

SiGaussianTailNoiseAdder::~SiGaussianTailNoiseAdder() {}

void SiGaussianTailNoiseAdder::addNoise(std::vector<float> &in,
                                        size_t &minChannel,
                                        size_t &maxChannel,
                                        int numStrips,
                                        float noiseRMS,
                                        CLHEP::HepRandomEngine *engine) const {
  std::vector<std::pair<int, float> > generatedNoise;
  genNoise->generate(numStrips, threshold, noiseRMS, generatedNoise, engine);

  // noise on strips with signal:
  for (size_t iChannel = minChannel; iChannel < maxChannel; iChannel++) {
    if (in[iChannel] != 0) {
      in[iChannel] += CLHEP::RandGaussQ::shoot(engine, 0., noiseRMS);
    }
  }

  // Noise on the other strips
  typedef std::vector<std::pair<int, float> >::const_iterator VI;
  for (VI p = generatedNoise.begin(); p != generatedNoise.end(); p++) {
    if (in[(*p).first] == 0) {
      in[(*p).first] += (*p).second;
    }
  }
}

void SiGaussianTailNoiseAdder::addNoiseVR(std::vector<float> &in,
                                          std::vector<float> &noiseRMS,
                                          CLHEP::HepRandomEngine *engine) const {
  // Add noise
  // Full Gaussian noise is added everywhere
  for (size_t iChannel = 0; iChannel != in.size(); iChannel++) {
    if (noiseRMS[iChannel] > 0.)
      in[iChannel] += CLHEP::RandGaussQ::shoot(engine, 0., noiseRMS[iChannel]);
  }
}

void SiGaussianTailNoiseAdder::addPedestals(std::vector<float> &in, std::vector<float> &ped) const {
  for (size_t iChannel = 0; iChannel != in.size(); iChannel++) {
    if (ped[iChannel] > 0.)
      in[iChannel] += ped[iChannel];
  }
}

void SiGaussianTailNoiseAdder::addCMNoise(std::vector<float> &in,
                                          float cmnRMS,
                                          std::vector<bool> &badChannels,
                                          CLHEP::HepRandomEngine *engine) const {
  int nAPVs = in.size() / 128;
  std::vector<float> CMNv;
  CMNv.reserve(nAPVs);
  for (int APVn = 0; APVn < nAPVs; ++APVn)
    CMNv.push_back(CLHEP::RandGaussQ::shoot(engine, 0., cmnRMS));
  for (size_t iChannel = 0; iChannel != in.size(); iChannel++) {
    if (!badChannels[iChannel])
      in[iChannel] += CMNv[(int)(iChannel / 128)];
  }
}

void SiGaussianTailNoiseAdder::addBaselineShift(std::vector<float> &in, std::vector<bool> &badChannels) const {
  size_t nAPVs = in.size() / 128;
  std::vector<float> vShift;
  double apvCharge, apvMult;

  size_t iChannel;
  for (size_t APVn = 0; APVn < nAPVs; ++APVn) {
    apvMult = 0;
    apvCharge = 0;
    for (iChannel = APVn * 128; iChannel != APVn * 128 + 128; ++iChannel) {
      if (in[iChannel] > 0) {
        ++apvMult;
        apvCharge += in[iChannel];
      }
      if (apvMult == 0)
        vShift.push_back(0);
      else
        vShift.push_back(apvCharge / apvMult);
    }
  }

  for (iChannel = 0; iChannel != in.size(); ++iChannel) {
    if (!badChannels[iChannel])
      in[iChannel] -= vShift[(int)(iChannel / 128)];
  }
}
