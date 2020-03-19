#ifndef SimGeneral_PreMixingModule_PreMixingPileupCopy_h
#define SimGeneral_PreMixingModule_PreMixingPileupCopy_h

/** \class PreMixingPileupCopy
 *
 * This class takes care of existing pileup information in the case of pre-mixing
 *
 * Originally from DataMixingModule, tailored further for premixing.
 *
 ************************************************************/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProducesCollector.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfoNew.h"

#include <vector>
#include <string>

namespace reco {
  class GenParticle;
}
class PileUpEventPrincipal;

namespace edm {
  class ModuleCallingContext;

  class PreMixingPileupCopy {
  public:
    PreMixingPileupCopy(const edm::ParameterSet& ps, edm::ProducesCollector, edm::ConsumesCollector&& iC);
    ~PreMixingPileupCopy() = default;

    float getTrueNumInteractions(PileUpEventPrincipal const& pep) const;

    void addPileupInfo(PileUpEventPrincipal const& pep);
    const std::vector<PileupSummaryInfo>& getPileupSummaryInfo() const { return pileupSummaryStorage_; }
    int getBunchSpacing() const { return bsStorage_; }
    void putPileupInfo(edm::Event& e);

  private:
    edm::InputTag pileupInfoInputTag_;    // InputTag for PileupSummaryInfo
    edm::InputTag bunchSpacingInputTag_;  // InputTag for bunch spacing int
    edm::InputTag cfPlaybackInputTag_;    // InputTag for CrossingFrame Playback information

    std::vector<edm::InputTag> genPUProtonsInputTags_;

    // deliver data from addPileupInfo() to getPileupInfo() and putPileupInfo()
    CrossingFramePlaybackInfoNew crossingFramePlaybackStorage_;
    std::vector<PileupSummaryInfo> pileupSummaryStorage_;
    int bsStorage_;

    std::vector<std::string> genPUProtons_labels_;
    std::vector<std::vector<reco::GenParticle> > genPUProtons_;

    bool foundPlayback_;
  };
}  // namespace edm

#endif
