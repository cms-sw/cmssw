#ifndef SiPixelDigitizer_h
#define SiPixelDigitizer_h

/** \class SiPixelDigitizer
 *
 * SiPixelDigitizer produces digis from SimHits
 * The real algorithm is in SiPixelDigitizerAlgorithm
 *
 * \author Michele Pioppi-INFN Perugia
 *
 * \version   Sep 26 2005  

 *
 ************************************************************/

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"

class MagneticField;
class PileUpEventPrincipal;
class PixelGeomDetUnit;
class PSimHit;
class SiPixelDigitizerAlgorithm;
class TrackerGeometry;

namespace CLHEP {
  class HepRandomEngine;
}

namespace cms {
  class SiPixelDigitizer : public DigiAccumulatorMixMod {
  public:
    explicit SiPixelDigitizer(const edm::ParameterSet& conf, edm::ProducesCollector, edm::ConsumesCollector& iC);

    ~SiPixelDigitizer() override;

    void initializeEvent(edm::Event const& e, edm::EventSetup const& c) override;
    void accumulate(edm::Event const& e, edm::EventSetup const& c) override;
    void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c, edm::StreamID const&) override;
    void finalizeEvent(edm::Event& e, edm::EventSetup const& c) override;

    virtual void beginJob() {}

    void StorePileupInformation(std::vector<int>& numInteractionList,
                                std::vector<int>& bunchCrossingList,
                                std::vector<float>& TrueInteractionList,
                                std::vector<edm::EventID>& eventInfoList,
                                int bunchSpacing) override {
      PileupInfo_ = std::make_unique<PileupMixingContent>(
          numInteractionList, bunchCrossingList, TrueInteractionList, eventInfoList, bunchSpacing);
    }

    PileupMixingContent* getEventPileupInfo() override { return PileupInfo_.get(); }

  private:
    void accumulatePixelHits(edm::Handle<std::vector<PSimHit> >,
                             size_t globalSimHitIndex,
                             const unsigned int tofBin,
                             edm::EventSetup const& c);

    bool firstInitializeEvent_;
    bool firstFinalizeEvent_;
    bool applyLateReweighting_;
    const bool store_SimHitEntryExitPoints_;
    bool makeDigiSimLinks_;
    std::unique_ptr<SiPixelDigitizerAlgorithm> _pixeldigialgo;
    /** @brief Offset to add to the index of each sim hit to account for which crossing it's in.
*
* I need to know what each sim hit index will be when the hits from all crossing frames are merged into
* one collection (assuming the MixingModule is configured to create the crossing frame for all sim hits).
* To do this I'll record how many hits were in each crossing, and then add that on to the index for a given
* hit in a given crossing. This assumes that the crossings are processed in the same order here as they are
* put into the crossing frame, which I'm pretty sure is true.<br/>
* The key is the name of the sim hit collection. */
    std::map<std::string, size_t> crossingSimHitIndexOffset_;

    typedef std::vector<std::string> vstring;
    const std::string hitsProducer;
    const vstring trackerContainers;
    const TrackerGeometry* pDD = nullptr;
    const MagneticField* pSetup = nullptr;
    std::map<unsigned int, PixelGeomDetUnit const*> detectorUnits;
    CLHEP::HepRandomEngine* randomEngine_ = nullptr;

    std::unique_ptr<PileupMixingContent> PileupInfo_;

    const bool pilotBlades;         // Default = false
    const int NumberOfEndcapDisks;  // Default = 2

    const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> pDDToken_;
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> pSetupToken_;

    // infrastructure to reject dead pixels as defined in db (added by F.Blekman)
  };
}  // namespace cms

#endif
