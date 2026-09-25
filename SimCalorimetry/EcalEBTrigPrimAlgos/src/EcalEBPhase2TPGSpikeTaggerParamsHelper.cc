///
/// \class EcalEBPhase2TPGSpikeTaggerParamsHelper
///
/// This class implements the ECAL spike tagger parameter interface.
/// Modifications here are easier since the object in the CondFormats
/// package is not changed.
///

#include <utility>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2TPGSpikeTaggerParamsHelper.h"

EcalEBPhase2TPGSpikeTaggerParamsHelper::EcalEBPhase2TPGSpikeTaggerParamsHelper() : EcalEBPhase2TPGSpikeTaggerParams() {
  nodes_.resize(NUM_NODES);
}

EcalEBPhase2TPGSpikeTaggerParamsHelper::EcalEBPhase2TPGSpikeTaggerParamsHelper(
    const EcalEBPhase2TPGSpikeTaggerParams &params)
    : EcalEBPhase2TPGSpikeTaggerParams(params) {
  nodes_.resize(NUM_NODES);
}

EcalEBPhase2TPGSpikeTaggerParamsHelper::EcalEBPhase2TPGSpikeTaggerParamsHelper(const edm::ParameterSet &config) {
  nodes_.resize(NUM_NODES);
  createFromPSet(config);
}

void EcalEBPhase2TPGSpikeTaggerParamsHelper::createFromPSet(const edm::ParameterSet &config) {
  version_ = 1;

  setFwVersion(config.getParameter<unsigned int>("fwVersion"));

  // algo parameters
  const auto algoConfigs = config.getParameter<std::vector<edm::ParameterSet>>("algoConfigs");
  for (const auto &algoConfig : algoConfigs) {
    const auto algo = algoConfig.getParameter<std::string>("algo");
    if (algo == "ld") {
      setPerCrystalSpikeTaggerParams(algoConfig.getParameter<std::vector<edm::ParameterSet>>("perCrystalParams"));
    } else {
      edm::LogError("EcalEBPhase2TPGSpikeTaggerParamsHelper") << "Unknown algorithm '" << algo << "'";
    }
  }
}

//////////////////////////////////////////////////////////////////////////////
// Global parameters
unsigned int EcalEBPhase2TPGSpikeTaggerParamsHelper::fwVersion() const {
  return nodes_[kGlobalAlgoParams].uparams_.size() > UIdx::kFwVersion
             ? nodes_[kGlobalAlgoParams].uparams_[UIdx::kFwVersion]
             : 0;
}

void EcalEBPhase2TPGSpikeTaggerParamsHelper::setFwVersion(const unsigned int fwVersion) {
  if (nodes_[kGlobalAlgoParams].uparams_.size() > UIdx::kFwVersion) {
    nodes_[kGlobalAlgoParams].uparams_[UIdx::kFwVersion] = fwVersion;
  } else {
    nodes_[kGlobalAlgoParams].uparams_.emplace_back(fwVersion);
  }
}

unsigned int EcalEBPhase2TPGSpikeTaggerParamsHelper::peakSampleIndex(const EBDetId &detId) const {
  const auto nodesIt = crystalNodes_.find(detId.rawId());
  return nodesIt->at(kCrystalSpikeTaggerLdParams).uparams_.size() > UIdx::kPeakSampleIndex
             ? nodesIt->at(kCrystalSpikeTaggerLdParams).uparams_[UIdx::kPeakSampleIndex]
             : 0;
}

void EcalEBPhase2TPGSpikeTaggerParamsHelper::setPeakSampleIndex(const EBDetId &detId, const unsigned int soi) {
  const auto rawId = detId.rawId();

  // make sure that all nodes exist
  crystalNodes_[rawId].resize(NUM_CRYSTAL_NODES);

  // set parameters for this crystal
  if (crystalNodes_[rawId][kCrystalSpikeTaggerLdParams].uparams_.size() > UIdx::kPeakSampleIndex) {
    crystalNodes_[rawId][kCrystalSpikeTaggerLdParams].uparams_[UIdx::kPeakSampleIndex] = soi;
  } else {
    crystalNodes_[rawId][kCrystalSpikeTaggerLdParams].uparams_.push_back(soi);
  }
}

//////////////////////////////////////////////////////////////////////////////
// Spike tagger LD parameters
double EcalEBPhase2TPGSpikeTaggerParamsHelper::spikeTaggerLdThreshold(const EBDetId &detId) const {
  const auto nodesIt = crystalNodes_.find(detId.rawId());
  return nodesIt->at(kCrystalSpikeTaggerLdParams).dparams_.size() > DIdx::kSpikeThreshold
             ? nodesIt->at(kCrystalSpikeTaggerLdParams).dparams_[DIdx::kSpikeThreshold]
             : 0.;
}

void EcalEBPhase2TPGSpikeTaggerParamsHelper::setSpikeTaggerLdThreshold(const EBDetId &detId, const double &thr) {
  const auto rawId = detId.rawId();

  // make sure that all nodes exist
  crystalNodes_[rawId].resize(NUM_CRYSTAL_NODES);

  // set parameters for this crystal
  if (crystalNodes_[rawId][kCrystalSpikeTaggerLdParams].dparams_.size() > DIdx::kSpikeThreshold) {
    crystalNodes_[rawId][kCrystalSpikeTaggerLdParams].dparams_[DIdx::kSpikeThreshold] = thr;
  } else {
    crystalNodes_[rawId][kCrystalSpikeTaggerLdParams].dparams_.push_back(thr);
  }
}

std::vector<double> EcalEBPhase2TPGSpikeTaggerParamsHelper::spikeTaggerLdWeights(const EBDetId &detId) const {
  const auto nodesIt = crystalNodes_.find(detId.rawId());
  return nodesIt->at(kCrystalSpikeTaggerLdWeights).dparams_;
}

void EcalEBPhase2TPGSpikeTaggerParamsHelper::setSpikeTaggerLdWeights(const EBDetId &detId,
                                                                     const std::vector<double> &weights) {
  const auto rawId = detId.rawId();

  // make sure that all nodes exist
  crystalNodes_[rawId].resize(NUM_CRYSTAL_NODES);

  // set parameters for this crystal
  crystalNodes_[rawId][kCrystalSpikeTaggerLdWeights].dparams_ = weights;
}

// print parameters to stream:
void EcalEBPhase2TPGSpikeTaggerParamsHelper::print(std::ostream &out) const {
  out << "ECAL spike tagger parameters" << std::endl;
  out << "Parameter version 0x" << std::hex << version_ << std::dec << std::endl;
  out << "Global parameters:" << std::endl;
  out << " Spike tagger firmware version 0x" << std::hex << this->fwVersion() << std::dec << std::endl;
  // TODO: output of per crystal parameters in usable format
  //out << "  Peak sample index " << this->peakSampleIndex() << std::endl;
  out << "Spike tagger LD parameters:" << std::endl;
  //out << "  Spike threshold " << this->spikeTaggerLdThreshold() << std::endl;
  //out << "  Polynomial weights (ascending order)" << std::endl;
  //for (const auto weight : this->spikeTaggerLdWeights()) {
  //  out << "    " << weight << std::endl;
  //}
}

std::ostream &operator<<(std::ostream &out, const EcalEBPhase2TPGSpikeTaggerParamsHelper &params) {
  params.print(out);
  return out;
}

void EcalEBPhase2TPGSpikeTaggerParamsHelper::setPerCrystalSpikeTaggerParams(
    const std::vector<edm::ParameterSet> &pSets) {
  for (const auto &pSet : pSets) {
    int ietaMin, ietaMax;
    int iphiMin, iphiMax;
    parseCrystalRange(pSet.getParameter<std::string>("ietaRange"), ietaMin, ietaMax);
    parseCrystalRange(pSet.getParameter<std::string>("iphiRange"), iphiMin, iphiMax, false);
    for (int ieta = ietaMin; ieta <= ietaMax; ++ieta) {
      // skip non-existing ieta == 0 crystals
      if (ieta == 0)
        continue;
      for (int iphi = iphiMin; iphi <= iphiMax; ++iphi) {
        if (EBDetId::validDetId(ieta, iphi)) {
          const EBDetId detId(ieta, iphi);
          this->setPeakSampleIndex(detId, pSet.getParameter<unsigned int>("peakSampleIndex"));
          this->setSpikeTaggerLdThreshold(detId, pSet.getParameter<double>("spikeThreshold"));
          this->setSpikeTaggerLdWeights(detId, pSet.getParameter<std::vector<double>>("weights"));
        }
      }
    }
  }
}

void EcalEBPhase2TPGSpikeTaggerParamsHelper::parseCrystalRange(const std::string &rangeStr,
                                                               int &iMin,
                                                               int &iMax,
                                                               const bool isEta) {
  const auto divPos = rangeStr.find(':');
  const auto minStr = rangeStr.substr(0, divPos);
  const auto maxStr = rangeStr.substr(divPos + 1);

  const int min = isEta ? -1 * EBDetId::MAX_IETA : EBDetId::MIN_IPHI;
  const int max = isEta ? EBDetId::MAX_IETA : EBDetId::MAX_IPHI;

  iMin = !minStr.empty() ? std::stoi(minStr) : min;
  iMax = !maxStr.empty() ? std::stoi(maxStr) : max;
}
