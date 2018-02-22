#include "PreMixingPileupCopy.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ProducerBase.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <memory>

namespace edm {
  PreMixingPileupCopy::PreMixingPileupCopy(const edm::ParameterSet& ps, edm::ProducerBase& producer, edm::ConsumesCollector && iC):
    pileupInfoInputTag_(ps.getParameter<edm::InputTag>("PileupInfoInputTag")),
    bunchSpacingInputTag_(ps.getParameter<edm::InputTag>("BunchSpacingInputTag")),
    cfPlaybackInputTag_(ps.getParameter<edm::InputTag>("CFPlaybackInputTag")),
    genPUProtonsInputTags_(ps.getParameter<std::vector<edm::InputTag> >("GenPUProtonsInputTags"))
  {
    // apparently, we don't need consumes from Secondary input stream
    //iC.consumes<std::vector<PileupSummaryInfo>>(PileupInfoInputTag_);
    //iC.consumes<int>(BunchSpacingInputTag_);
    //iC.consumes<CrossingFramePlaybackInfoNew>(CFPlaybackInputTag_);

    producer.produces<std::vector<PileupSummaryInfo> >();
    producer.produces<int>("bunchSpacing");
    producer.produces<CrossingFramePlaybackInfoNew>();

    for(const auto& tag: genPUProtonsInputTags_) {
      producer.produces<std::vector<reco::GenParticle> >(tag.label());
    }
  }
	       
  void PreMixingPileupCopy::addPileupInfo(const EventPrincipal& ep, unsigned int eventNr, ModuleCallingContext const* mcc) {
  
    LogDebug("PreMixingPileupCopy") <<"\n===============> adding pileup Info from event  "<<ep.id();

    // find PileupSummaryInfo, CFPlayback information, if it's there

    // Pileup info first

    std::shared_ptr<Wrapper< std::vector<PileupSummaryInfo> >  const> pileupInfoPTR =
      getProductByTag<std::vector<PileupSummaryInfo>>(ep, pileupInfoInputTag_, mcc);

    std::shared_ptr<Wrapper< int >  const> bsPTR =
      getProductByTag<int>(ep,bunchSpacingInputTag_, mcc);

    if(pileupInfoPTR) {
      pileupSummaryStorage_ = *(pileupInfoPTR->product());
      LogDebug("PreMixingPileupCopy") << "PileupInfo Size: " << pileupSummaryStorage_.size();
    }
    bsStorage_ = bsPTR ? *(bsPTR->product()) : 10000;

    // Gen. PU protons
    std::shared_ptr<edm::Wrapper<std::vector<reco::GenParticle> > const> genPUProtonsPTR;
    for(const auto& tag: genPUProtonsInputTags_) {
      genPUProtonsPTR = getProductByTag<std::vector<reco::GenParticle> >(ep, tag, mcc);
      if(genPUProtonsPTR != nullptr) {
        genPUProtons_.push_back(*(genPUProtonsPTR->product()));
         genPUProtons_labels_.push_back(tag.label());
      }
      else {
        edm::LogWarning("PreMixingPileupCopy") << "Missing product with label: " << tag.label();
      }
    }

    // Playback
    std::shared_ptr<Wrapper<CrossingFramePlaybackInfoNew>  const> playbackPTR =
      getProductByTag<CrossingFramePlaybackInfoNew>(ep,cfPlaybackInputTag_, mcc);
    foundPlayback_ = false;
    if(playbackPTR ) {
      crossingFramePlaybackStorage_ = *(playbackPTR->product());
      foundPlayback_ = true;
    }
  }
 
  void PreMixingPileupCopy::putPileupInfo(edm::Event &e) {
    if(foundPlayback_ ) {
      e.put(std::make_unique<CrossingFramePlaybackInfoNew>(std::move(crossingFramePlaybackStorage_)));
    }
    e.put(std::make_unique<std::vector<PileupSummaryInfo> >(std::move(pileupSummaryStorage_)));
    e.put(std::make_unique<int>(bsStorage_), "bunchSpacing");

    // Gen. PU protons
    for(size_t idx = 0; idx < genPUProtons_.size(); ++idx){
      e.put(std::make_unique<std::vector<reco::GenParticle> >(std::move(genPUProtons_[idx])),
            genPUProtons_labels_[idx]);
    }

    // clear local storage after this event
    pileupSummaryStorage_.clear();
    genPUProtons_.clear();
    genPUProtons_labels_.clear();
  }
} //edm
