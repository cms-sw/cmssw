#ifndef __SimTracker_SiPhase2Digitizer_Phase2TrackerDigitizer_h
#define __SimTracker_SiPhase2Digitizer_Phase2TrackerDigitizer_h

//-------------------------------------------------------------
// class Phase2TrackerDigitizer
//
// Phase2TrackerDigitizer produces digis from SimHits
// The real algorithm is in Phase2TrackerDigitizerAlgorithm
//
// Author: Suchandra Dutta, Suvankar Roy Chowdhury, Subir Sarkar
//
//--------------------------------------------------------------

#include <map>
#include <string>
#include <vector>
#include <unordered_map>

#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerFwd.h"

// Forward declaration
namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
  template <typename T>
  class Handle;
  class ConsumesCollector;
}  // namespace edm

class MagneticField;
class PileUpEventPrincipal;
class PSimHit;
class Phase2TrackerDigitizerAlgorithm;
class TrackerDigiGeometryRecord;

namespace cms {
  class Phase2TrackerDigitizer : public DigiAccumulatorMixMod {
  public:
    using ModuleTypeCache = std::unordered_map<uint32_t, TrackerGeometry::ModuleType>;

    explicit Phase2TrackerDigitizer(const edm::ParameterSet& iConfig,
                                    edm::ProducesCollector,
                                    edm::ConsumesCollector& iC);
    ~Phase2TrackerDigitizer() override;

    void initializeEvent(edm::Event const& e, edm::EventSetup const& c) override;
    void accumulate(edm::Event const& e, edm::EventSetup const& c) override;
    void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c, edm::StreamID const&) override;
    void finalizeEvent(edm::Event& e, edm::EventSetup const& c) override;
    virtual void beginJob() {}

    template <class T>
    void accumulate_local(T const& iEvent, edm::EventSetup const& iSetup);

    // For premixing
    void loadAccumulator(const std::map<uint32_t, std::map<int, float> >& accumulator);

  private:
    using vstring = std::vector<std::string>;

    // constants of different algorithm types
    enum class AlgorithmType { InnerPixel, InnerPixel3D, PixelinPS, StripinPS, TwoStrip, Unknown };
    AlgorithmType getAlgoType(uint32_t idet);

    void accumulatePixelHits(edm::Handle<std::vector<PSimHit> >, size_t globalSimHitIndex, const uint32_t tofBin);
    void addPixelCollection(edm::Event& iEvent, const edm::EventSetup& iSetup, const bool ot_analog);

    // Templated for premixing
    template <typename DigiType>
    void addOuterTrackerCollection(edm::Event& iEvent, const edm::EventSetup& iSetup);

    bool first_;

    /** @brief Offset to add to the index of each sim hit to account for which crossing it's in.
     *
     * I need to know what each sim hit index will be when the hits from all crossing frames are merged into
     * one collection (assuming the MixingModule is configured to create the crossing frame for all sim hits).
     * To do this I'll record how many hits were in each crossing, and then add that on to the index for a given
     * hit in a given crossing. This assumes that the crossings are processed in the same order here as they are
     * put into the crossing frame, which I'm pretty sure is true.<br/>
     * The key is the name of the sim hit collection. */
    std::map<std::string, size_t> crossingSimHitIndexOffset_;
    std::map<AlgorithmType, std::unique_ptr<Phase2TrackerDigitizerAlgorithm> > algomap_;
    const std::string hitsProducer_;
    const vstring trackerContainers_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> pDDToken_;
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> pSetupToken_;
    const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
    const TrackerGeometry* pDD_ = nullptr;
    const MagneticField* pSetup_ = nullptr;
    std::map<uint32_t, const Phase2TrackerGeomDetUnit*> detectorUnits_;
    const TrackerTopology* tTopo_ = nullptr;
    edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher_;
    const bool isOuterTrackerReadoutAnalog_;
    const bool premixStage1_;
    const bool makeDigiSimLinks_;
    // cache for detector types
    ModuleTypeCache moduleTypeCache_;
  };
}  // namespace cms
#endif
