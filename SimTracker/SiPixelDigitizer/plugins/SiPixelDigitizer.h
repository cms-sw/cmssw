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

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Provenance/interface/EventID.h"

namespace edm {
  class ConsumesCollector;
  namespace one {
    class EDProducerBase;
  }
  class Event;
  class EventSetup;
  class ParameterSet;
  template<typename T> class Handle;
  class StreamID;
}

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

    explicit SiPixelDigitizer(const edm::ParameterSet& conf, edm::one::EDProducerBase& mixMod, edm::ConsumesCollector& iC);

    virtual ~SiPixelDigitizer();

    virtual void initializeEvent(edm::Event const& e, edm::EventSetup const& c) override;
    virtual void accumulate(edm::Event const& e, edm::EventSetup const& c) override;
    virtual void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c, edm::StreamID const&) override;
    virtual void finalizeEvent(edm::Event& e, edm::EventSetup const& c) override;

    virtual void beginJob() {}

    virtual void StorePileupInformation( std::vector<int> &numInteractionList,
					 std::vector<int> &bunchCrossingList,
					 std::vector<float> &TrueInteractionList, 
					 std::vector<edm::EventID> &eventInfoList, int bunchSpacing){
      PileupInfo_ = new PileupMixingContent(numInteractionList, bunchCrossingList, TrueInteractionList, eventInfoList, bunchSpacing);
    }

    virtual PileupMixingContent* getEventPileupInfo() { return PileupInfo_; }

  private:
    void accumulatePixelHits(edm::Handle<std::vector<PSimHit> >,
			     size_t globalSimHitIndex,
			     const unsigned int tofBin,
			     CLHEP::HepRandomEngine*,
			     edm::EventSetup const& c);
    CLHEP::HepRandomEngine* randomEngine(edm::StreamID const& streamID);

    bool firstInitializeEvent_;
    bool firstFinalizeEvent_;
    std::unique_ptr<SiPixelDigitizerAlgorithm>  _pixeldigialgo;
    /** @brief Offset to add to the index of each sim hit to account for which crossing it's in.
*
* I need to know what each sim hit index will be when the hits from all crossing frames are merged into
* one collection (assuming the MixingModule is configured to create the crossing frame for all sim hits).
* To do this I'll record how many hits were in each crossing, and then add that on to the index for a given
* hit in a given crossing. This assumes that the crossings are processed in the same order here as they are
* put into the crossing frame, which I'm pretty sure is true.<br/>
* The key is the name of the sim hit collection. */
    std::map<std::string,size_t> crossingSimHitIndexOffset_;

    typedef std::vector<std::string> vstring;
    const std::string hitsProducer;
    const vstring trackerContainers;
    const std::string geometryType;
    edm::ESHandle<TrackerGeometry> pDD;
    edm::ESHandle<MagneticField> pSetup;
    std::map<unsigned int, PixelGeomDetUnit const *> detectorUnits;
    std::vector<CLHEP::HepRandomEngine*> randomEngines_;

    PileupMixingContent* PileupInfo_;
    
    const bool pilotBlades; // Default = false
    const int NumberOfEndcapDisks; // Default = 2
    
    // infrastructure to reject dead pixels as defined in db (added by F.Blekman)
  };
}


#endif
