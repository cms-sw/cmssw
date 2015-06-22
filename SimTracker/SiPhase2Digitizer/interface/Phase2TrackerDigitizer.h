#ifndef __SimTracker_SiPhase2Digitizer_Phase2TrackerDigitizer_h
#define __SimTracker_SiPhase2Digitizer_Phase2TrackerDigitizer_h

//-------------------------------------------------------------
// class Phase2TrackerDigitizer
//
// Phase2TrackerDigitizer produces digis from SimHits
// The real algorithm is in Phase2TrackerDigitizerAlgorithm
//
// version 1.1 August 18 2014  
//
//--------------------------------------------------------------

#include <map>
#include <string>
#include <vector>

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"

#include "SimTracker/SiPhase2Digitizer/interface/Phase2TrackerDigitizerFwd.h"

// Forward declaration
namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {
  class EDProducer;
  class Event;
  class EventSetup;
  class ParameterSet;
  template<typename T> class Handle;
}

class MagneticField;
class PileUpEventPrincipal;
class PSimHit;
class Phase2TrackerDigitizerAlgorithm;
class TrackerGeometry;

namespace cms 
{
  class Phase2TrackerDigitizer: public DigiAccumulatorMixMod {

  public:
    explicit Phase2TrackerDigitizer(const edm::ParameterSet& conf, edm::EDProducer& mixMod);
    virtual ~Phase2TrackerDigitizer();
    virtual void initializeEvent(edm::Event const& e, edm::EventSetup const& c) override;
    virtual void accumulate(edm::Event const& e, edm::EventSetup const& c) override;
    virtual void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c) override;
    virtual void finalizeEvent(edm::Event& e, edm::EventSetup const& c) override;
    virtual void beginJob() {}
    void beginRun(edm::Run const& run, edm::EventSetup const& iSetup);
    std::string getAlgoType(unsigned int idet); 
    template <class T>
    void accumulate_local(T const& iEvent, edm::EventSetup const& iSetup);

    // constants of different algorithm types
    const static std::string InnerPixel;
    const static std::string PixelinPS;
    const static std::string StripinPS;
    const static std::string TwoStrip;        
  
  private:
    typedef std::vector<std::string> vstring;

    void accumulatePixelHits(edm::Handle<std::vector<PSimHit> >, 
			     size_t globalSimHitIndex,
			     const unsigned int tofBin);   
    void addPixelCollection(edm::Event& iEvent, const edm::EventSetup& iSetup);
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
    std::map<std::string, std::unique_ptr<Phase2TrackerDigitizerAlgorithm> > algomap_;
    const std::string hitsProducer_;
    const vstring trackerContainers_;
    const std::string geometryType_;
    edm::ESHandle<TrackerGeometry> pDD_;
    edm::ESHandle<MagneticField> pSetup_;
    std::map<unsigned int, Phase2TrackerGeomDetUnit*> detectorUnits_;
    CLHEP::HepRandomEngine* rndEngine_;
    const StackedTrackerGeometry* stkGeom_;
    std::map<DetId, StackedTrackerDetUnit*> detIdStackDetIdmap_;
    edm::ESHandle<TrackerTopology> tTopoHand;

  };
}
#endif
