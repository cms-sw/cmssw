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
				 std::vector<float> &TrueInteractionList){
      PileupInfo_ = new PileupMixingContent(numInteractionList, bunchCrossingList, TrueInteractionList);
    }

    virtual PileupMixingContent* getEventPileupInfo() { return PileupInfo_; }

  private:
    void accumulatePixelHits(edm::Handle<std::vector<PSimHit> >, CLHEP::HepRandomEngine*);
    CLHEP::HepRandomEngine* randomEngine(edm::StreamID const& streamID);

    bool first;
    std::unique_ptr<SiPixelDigitizerAlgorithm>  _pixeldigialgo;
    typedef std::vector<std::string> vstring;
    const std::string hitsProducer;
    const vstring trackerContainers;
    const std::string geometryType;
    edm::ESHandle<TrackerGeometry> pDD;
    edm::ESHandle<MagneticField> pSetup;
    std::map<unsigned int, PixelGeomDetUnit*> detectorUnits;
    std::vector<CLHEP::HepRandomEngine*> randomEngines_;

    PileupMixingContent* PileupInfo_;

    // infrastructure to reject dead pixels as defined in db (added by F.Blekman)
  };
}


#endif
