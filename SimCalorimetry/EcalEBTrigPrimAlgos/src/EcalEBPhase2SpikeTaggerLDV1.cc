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
  peakIdx_ = spikeTaggerParamsHelper_->peakSampleIndex(detId);
  spikeThreshold_ = spikeTaggerParamsHelper_->spikeTaggerLdThreshold(detId);
  weights_ = spikeTaggerParamsHelper_->spikeTaggerLdWeights(detId);

  LogDebug("EcalEBPhase2SpikeTaggerLDV1").log([&](auto& lm) {
    lm << "Set parameters for channel at " << detId.ieta() << "," << detId.iphi()
       << " (ieta,iphi). peak index: " << peakIdx_ << ", spike threshold: " << spikeThreshold_
       << ", weights (ascending order):";
    for (auto const weight : weights_) {
      lm << " " << weight;
    }
  });
}

bool EcalEBPhase2SpikeTaggerLDV1::process(const std::vector<int>& linInput) {
  // need to be able to access a sample before and after the peak sample
  if (peakIdx_ == 0 || peakIdx_ > linInput.size() - 2) {
    throw cms::Exception("IndexOutOfBounds")
        << "Index of peak sample (" << peakIdx_ << ") for LD spike tagger is outside of allowed values (0 < idx < "
        << linInput.size() - 1 << ").";
  }

  // calculate the LD variable with as many polynomial terms as there are weights
  auto const ld = calcLD(linInput);

  LogDebug("EcalEBPhase2SpikeTaggerLDV1").log([&](auto& lm) {
    lm << "linearized digis: ";
    for (auto const& linIn : linInput) {
      lm << " " << linIn;
    }
    lm << ", LD value: " << ld << ", spike: " << (ld < spikeThreshold_);
  });

  // calculate and return LD spike flag
  return ld < spikeThreshold_;
}

float EcalEBPhase2SpikeTaggerLDV1::calcLD(std::vector<int> const& linInput) const {
  if (linInput[peakIdx_] == 0)
    return 0.;

  auto const sMax = static_cast<float>(linInput[peakIdx_]);
  auto const sPlus1 = linInput[peakIdx_ + 1];
  auto const rPlus1 = sPlus1 / sMax;

  return rPlus1 - calcRMinus1Poly(linInput, sMax);
}

float EcalEBPhase2SpikeTaggerLDV1::calcRMinus1Poly(std::vector<int> const& linInput, float sMax) const {
  auto const sMinus1 = linInput[peakIdx_ - 1];
  auto const rMinus1 = sMinus1 / sMax;
  float rMinus1Pow = 1.;
  float rMinus1Poly = 0.;
  for (auto const weight : weights_) {
    rMinus1Poly += weight * rMinus1Pow;
    rMinus1Pow *= rMinus1;
  }

  return rMinus1Poly;
}
