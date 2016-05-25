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

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerFwd.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/stream/EDProducerBase.h"

// Forward declaration
namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
  template<typename T> class Handle;
  class ConsumesCollector;
}

class MagneticField;
class PileUpEventPrincipal;
class PSimHit;
class Phase2TrackerDigitizerAlgorithm;
class TrackerGeometry;
class TrackerDigiGeometryRecord;

namespace cms 
{
  class Phase2TrackerDigitizer: public DigiAccumulatorMixMod {

  public:
    explicit Phase2TrackerDigitizer(const edm::ParameterSet& iConfig, edm::stream::EDProducerBase& mixMod, edm::ConsumesCollector& iC);
    virtual ~Phase2TrackerDigitizer();
    virtual void initializeEvent(edm::Event const& e, edm::EventSetup const& c) override;
    virtual void accumulate(edm::Event const& e, edm::EventSetup const& c) override;
    virtual void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c, edm::StreamID const&) override;
    virtual void finalizeEvent(edm::Event& e, edm::EventSetup const& c) override;
    virtual void beginJob() {}
    virtual void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& iSetup) override;
    virtual void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& iSetup) override; 

    template <class T>
    void accumulate_local(T const& iEvent, edm::EventSetup const& iSetup);

  
  private:
    using vstring = std::vector<std::string> ;

    // constants of different algorithm types
    enum class AlgorithmType {
      InnerPixel,
      PixelinPS,
      StripinPS,
      TwoStrip,
      Unknown   
    };
    AlgorithmType getAlgoType(unsigned int idet); 

    void accumulatePixelHits(edm::Handle<std::vector<PSimHit> >, 
			     size_t globalSimHitIndex,
			     const unsigned int tofBin);   
    void addPixelCollection(edm::Event& iEvent, const edm::EventSetup& iSetup, const bool ot_analog);
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
    std::map<std::string,size_t> crossingSimHitIndexOffset_; 
    std::map<AlgorithmType, std::unique_ptr<Phase2TrackerDigitizerAlgorithm> > algomap_;
    const std::string hitsProducer_;
    const vstring trackerContainers_;
    const std::string geometryType_;
    edm::ESHandle<TrackerGeometry> pDD_;
    edm::ESHandle<MagneticField> pSetup_;
    std::map<unsigned int, const Phase2TrackerGeomDetUnit*> detectorUnits_;
    CLHEP::HepRandomEngine* rndEngine_;
    edm::ESHandle<TrackerTopology> tTopoHand;
    edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;
    const edm::ParameterSet& iconfig_;

  };
}
#endif
