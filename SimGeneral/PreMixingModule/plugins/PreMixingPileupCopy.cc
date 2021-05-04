#include "PreMixingPileupCopy.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <memory>

namespace edm {
  PreMixingPileupCopy::PreMixingPileupCopy(const edm::ParameterSet& ps,
                                           edm::ProducesCollector producesCollector,
                                           edm::ConsumesCollector&& iC)
      : pileupInfoInputTag_(ps.getParameter<edm::InputTag>("PileupInfoInputTag")),
        bunchSpacingInputTag_(ps.getParameter<edm::InputTag>("BunchSpacingInputTag")),
        cfPlaybackInputTag_(ps.getParameter<edm::InputTag>("CFPlaybackInputTag")),
        genPUProtonsInputTags_(ps.getParameter<std::vector<edm::InputTag>>("GenPUProtonsInputTags")) {
    producesCollector.produces<std::vector<PileupSummaryInfo>>();
    producesCollector.produces<int>("bunchSpacing");
    producesCollector.produces<CrossingFramePlaybackInfoNew>();

    for (const auto& tag : genPUProtonsInputTags_) {
      producesCollector.produces<std::vector<reco::GenParticle>>(tag.label());
    }
  }

  float PreMixingPileupCopy::getTrueNumInteractions(PileUpEventPrincipal const& pep) const {
    edm::Handle<std::vector<PileupSummaryInfo>> pileupInfoHandle;
    pep.getByLabel(pileupInfoInputTag_, pileupInfoHandle);

    auto it = std::find_if(
        pileupInfoHandle->begin(), pileupInfoHandle->end(), [](const auto& s) { return s.getBunchCrossing() == 0; });
    if (it == pileupInfoHandle->end()) {
      throw cms::Exception("LogicError") << "Did not find PileupSummaryInfo in bunch crossing 0";
    }

    return it->getTrueNumInteractions();
  }

  void PreMixingPileupCopy::addPileupInfo(const PileUpEventPrincipal& pep) {
    LogDebug("PreMixingPileupCopy") << "\n===============> adding pileup Info from event  " << pep.principal().id();

    // find PileupSummaryInfo, CFPlayback information, if it's there

    // Pileup info first
    edm::Handle<std::vector<PileupSummaryInfo>> pileupInfoHandle;
    pep.getByLabel(pileupInfoInputTag_, pileupInfoHandle);

    edm::Handle<int> bsHandle;
    pep.getByLabel(bunchSpacingInputTag_, bsHandle);

    if (pileupInfoHandle.isValid()) {
      pileupSummaryStorage_ = *pileupInfoHandle;
      LogDebug("PreMixingPileupCopy") << "PileupInfo Size: " << pileupSummaryStorage_.size();
    }
    bsStorage_ = bsHandle.isValid() ? *bsHandle : 10000;

    // Gen. PU protons
    edm::Handle<std::vector<reco::GenParticle>> genPUProtonsHandle;
    for (const auto& tag : genPUProtonsInputTags_) {
      pep.getByLabel(tag, genPUProtonsHandle);
      if (genPUProtonsHandle.isValid()) {
        genPUProtons_.push_back(*genPUProtonsHandle);
        genPUProtons_labels_.push_back(tag.label());
      } else {
        edm::LogWarning("PreMixingPileupCopy") << "Missing product with label: " << tag.label();
      }
    }

    // Playback
    edm::Handle<CrossingFramePlaybackInfoNew> playbackHandle;
    pep.getByLabel(cfPlaybackInputTag_, playbackHandle);
    foundPlayback_ = false;
    if (playbackHandle.isValid()) {
      crossingFramePlaybackStorage_ = *playbackHandle;
      foundPlayback_ = true;
    }
  }

  void PreMixingPileupCopy::putPileupInfo(edm::Event& e) {
    if (foundPlayback_) {
      e.put(std::make_unique<CrossingFramePlaybackInfoNew>(std::move(crossingFramePlaybackStorage_)));
    }
    e.put(std::make_unique<std::vector<PileupSummaryInfo>>(std::move(pileupSummaryStorage_)));
    e.put(std::make_unique<int>(bsStorage_), "bunchSpacing");

    // Gen. PU protons
    for (size_t idx = 0; idx < genPUProtons_.size(); ++idx) {
      e.put(std::make_unique<std::vector<reco::GenParticle>>(std::move(genPUProtons_[idx])), genPUProtons_labels_[idx]);
    }

    // clear local storage after this event
    pileupSummaryStorage_.clear();
    genPUProtons_.clear();
    genPUProtons_labels_.clear();
  }
}  // namespace edm
