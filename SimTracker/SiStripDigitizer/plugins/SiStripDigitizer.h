#ifndef SiStripDigitizer_h
#define SiStripDigitizer_h

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <bitset>
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvSimulationParameters.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupMixingContent.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"

class TrackerTopology;

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {
  class ConsumesCollector;
  class Event;
  class EventSetup;
  class ParameterSet;
  template <typename T>
  class Handle;
  class StreamID;
}  // namespace edm

class MagneticField;
class PileUpEventPrincipal;
class PSimHit;
class SiStripDigitizerAlgorithm;
class StripGeomDetUnit;
class TrackerGeometry;
class SiStripBadStrip;

/** @brief Accumulator to perform digitisation on the strip tracker sim hits.
 *
 * @author original author unknown; converted from a producer to a MixingModule accumulator by Bill
 * Tanenbaum; functionality to create digi-sim links moved from Bill's DigiSimLinkProducer into here
 * by Mark Grimes (mark.grimes@bristol.ac.uk).
 * @date original date unknown; moved into a MixingModule accumulator mid to late 2012; digi sim links
 * eventually finished May 2013
 */
class SiStripDigitizer : public DigiAccumulatorMixMod {
public:
  explicit SiStripDigitizer(const edm::ParameterSet& conf, edm::ProducesCollector, edm::ConsumesCollector& iC);

  ~SiStripDigitizer() override;

  void initializeEvent(edm::Event const& e, edm::EventSetup const& c) override;
  void accumulate(edm::Event const& e, edm::EventSetup const& c) override;
  void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c, edm::StreamID const&) override;
  void finalizeEvent(edm::Event& e, edm::EventSetup const& c) override;

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
  void accumulateStripHits(edm::Handle<std::vector<PSimHit>>,
                           const TrackerTopology* tTopo,
                           size_t globalSimHitIndex,
                           const unsigned int tofBin);

  typedef std::vector<std::string> vstring;
  typedef std::map<unsigned int, std::vector<std::pair<const PSimHit*, int>>, std::less<unsigned int>> simhit_map;
  typedef simhit_map::iterator simhit_map_iterator;

  const std::string hitsProducer;
  const vstring trackerContainers;
  const std::string ZSDigi;
  const std::string SCDigi;
  const std::string VRDigi;
  const std::string PRDigi;
  const bool useConfFromDB;
  const bool zeroSuppression;
  const bool makeDigiSimLinks_;
  const bool includeAPVSimulation_;
  const double fracOfEventsToSimAPV_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> pDDToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> pSetupToken_;
  const edm::ESGetToken<SiStripGain, SiStripGainSimRcd> gainToken_;
  const edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noiseToken_;
  const edm::ESGetToken<SiStripThreshold, SiStripThresholdRcd> thresholdToken_;
  const edm::ESGetToken<SiStripPedestals, SiStripPedestalsRcd> pedestalToken_;
  const edm::ESGetToken<SiStripBadStrip, SiStripBadChannelRcd> deadChannelToken_;
  edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> detCablingToken_;
  edm::ESGetToken<SiStripApvSimulationParameters, SiStripApvSimulationParametersRcd> apvSimulationParametersToken_;

  unsigned long long ddCacheID_ = 0;
  unsigned long long deadChannelCacheID_ = 0;

  ///< Whether or not to create the association to sim truth collection. Set in configuration.
  /** @brief Offset to add to the index of each sim hit to account for which crossing it's in.
   *
   * I need to know what each sim hit index will be when the hits from all crossing frames are merged into
   * one collection (assuming the MixingModule is configured to create the crossing frame for all sim hits).
   * To do this I'll record how many hits were in each crossing, and then add that on to the index for a given
   * hit in a given crossing. This assumes that the crossings are processed in the same order here as they are
   * put into the crossing frame, which I'm pretty sure is true.<br/>
   * The key is the name of the sim hit collection. */
  std::map<std::string, size_t> crossingSimHitIndexOffset_;

  std::unique_ptr<SiStripDigitizerAlgorithm> theDigiAlgo;
  std::map<uint32_t, std::vector<int>> theDetIdList;
  const TrackerGeometry* pDD = nullptr;
  const MagneticField* pSetup = nullptr;
  std::map<unsigned int, StripGeomDetUnit const*> detectorUnits;
  CLHEP::HepRandomEngine* randomEngine_ = nullptr;
  std::vector<std::pair<int, std::bitset<6>>> theAffectedAPVvector;

  std::unique_ptr<PileupMixingContent> PileupInfo_;
};

#endif
