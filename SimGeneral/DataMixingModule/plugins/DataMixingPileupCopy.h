#ifndef SimDataMixingPileupCopy_h
#define SimDataMixingPileupCopy_h

/** \class DataMixingPileupCopy
 *
 * DataMixingModule is the EDProducer subclass
 * that overlays rawdata events on top of MC,
 * using real data for pileup simulation
 * This class takes care of existing pileup information in the case of
 *pre-mixing
 *
 * \author Mike Hildreth, University of Notre Dame
 *
 * \version   1st Version October 2007
 *
 ************************************************************/

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfoNew.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include <map>
#include <string>
#include <vector>

namespace edm {
  class ModuleCallingContext;

  class DataMixingPileupCopy {
  public:
    DataMixingPileupCopy();

    /** standard constructor*/
    explicit DataMixingPileupCopy(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC);

    /**Default destructor*/
    virtual ~DataMixingPileupCopy();

    void putPileupInfo(edm::Event &e);
    void addPileupInfo(const edm::EventPrincipal *, unsigned int EventId, ModuleCallingContext const *mcc);

    void getPileupInfo(std::vector<PileupSummaryInfo> &ps, int &bs) {
      ps = PileupSummaryStorage_;
      bs = bsStorage_;
    }

  private:
    // data specifiers

    edm::InputTag PileupInfoInputTag_;    // InputTag for PileupSummaryInfo
    edm::InputTag BunchSpacingInputTag_;  // InputTag for bunch spacing int
    edm::InputTag CFPlaybackInputTag_;    // InputTag for CrossingFrame Playback information

    std::vector<edm::InputTag> GenPUProtonsInputTags_;

    CrossingFramePlaybackInfoNew CrossingFramePlaybackStorage_;

    std::vector<PileupSummaryInfo> PileupSummaryStorage_;
    int bsStorage_;

    std::vector<std::string> GenPUProtons_labels_;
    std::vector<std::vector<reco::GenParticle>> GenPUProtons_;

    //      unsigned int eventId_; //=0 for signal, from 1-n for pileup events

    std::string label_;

    bool FoundPlayback_;
  };
}  // namespace edm

#endif  // SimDataMixingPileupCopy_h
