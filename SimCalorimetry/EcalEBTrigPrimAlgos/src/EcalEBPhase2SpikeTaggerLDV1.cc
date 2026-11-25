#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2SpikeTaggerLDV1.h"

EcalEBPhase2SpikeTaggerLDV1::EcalEBPhase2SpikeTaggerLDV1(edm::ConsumesCollector& cc, bool debug)
    : EcalEBPhase2SpikeTagger(cc, debug), spikeTaggerParamsToken_(cc.esConsumes<edm::Transition::BeginRun>()) {}

void EcalEBPhase2SpikeTaggerLDV1::getRecords(edm::EventSetup const& setup) {
  auto const& spikeTaggerParams = setup.getData(spikeTaggerParamsToken_);
  spikeTaggerParamsHelper_ = std::make_shared<EcalEBPhase2TPGSpikeTaggerParamsHelper>(spikeTaggerParams);
}

void EcalEBPhase2SpikeTaggerLDV1::setParameters(EBDetId detId, const EcalTPGCrystalStatus* ecaltpBadX) {
  bx0PeakIdx_ = spikeTaggerParamsHelper_->peakSampleIndex(detId);
  spikeThreshold_ = spikeTaggerParamsHelper_->spikeTaggerLdThreshold(detId);
  weights_ = spikeTaggerParamsHelper_->spikeTaggerLdWeights(detId);

  LogDebug("EcalEBPhase2SpikeTaggerLDV1").log([&](auto& lm) {
    lm << "Set parameters for channel at " << detId.ieta() << "," << detId.iphi()
       << " (ieta,iphi). BX0 peak index: " << bx0PeakIdx_ << ", spike threshold: " << spikeThreshold_
       << ", weights (ascending order):";
    for (auto const weight : weights_) {
      lm << " " << weight;
    }
  });
}

void EcalEBPhase2SpikeTaggerLDV1::process(const std::vector<int>& linInput, std::vector<bool>& spikeTags) {
  // need to be able to access a sample before and after the BX0 peak sample to calculate the LD spike tag for BX0
  if (bx0PeakIdx_ == 0 || bx0PeakIdx_ > linInput.size() - 2) {
    throw cms::Exception("IndexOutOfBounds")
        << "Index of BX0 peak sample (" << bx0PeakIdx_
        << ") for LD spike tagger is outside of allowed values (0 < idx < " << linInput.size() - 1 << ").";
  }

  // calulate the spike tag for BX0 and as many BX>0 as the input allows
  unsigned int tagIdx = 0;
  auto peakIdx = bx0PeakIdx_;
  while (peakIdx < linInput.size() - 2 and tagIdx < spikeTags.size()) {
    // calculate the LD variable with as many polynomial terms as there are weights
    auto const ld = calcLD(linInput, peakIdx);

    LogDebug("EcalEBPhase2SpikeTaggerLDV1").log([&](auto& lm) {
      lm << "linearized digis: ";
      for (auto const& linIn : linInput) {
        lm << " " << linIn;
      }
      lm << ", BX " << tagIdx << ", LD value: " << ld << ", spike: " << (ld < spikeThreshold_);
    });

    // calculate LD spike tag for the BX
    spikeTags[tagIdx] = ld < spikeThreshold_;

    ++tagIdx;
    peakIdx += 4;  // 160 MHz sampling rate. 4 samples per BX
  }
}

float EcalEBPhase2SpikeTaggerLDV1::calcLD(std::vector<int> const& linInput, unsigned int peakIdx) const {
  if (linInput[peakIdx] == 0)
    return 0.;

  auto const sMax = static_cast<float>(linInput[peakIdx]);
  auto const sPlus1 = linInput[peakIdx + 1];
  auto const rPlus1 = sPlus1 / sMax;

  return rPlus1 - calcRMinus1Poly(linInput, sMax, peakIdx);
}

float EcalEBPhase2SpikeTaggerLDV1::calcRMinus1Poly(std::vector<int> const& linInput,
                                                   float sMax,
                                                   unsigned int peakIdx) const {
  auto const sMinus1 = linInput[peakIdx - 1];
  auto const rMinus1 = sMinus1 / sMax;
  float rMinus1Pow = 1.;
  float rMinus1Poly = 0.;
  for (auto const weight : weights_) {
    rMinus1Poly += weight * rMinus1Pow;
    rMinus1Pow *= rMinus1;
  }

  return rMinus1Poly;
}
