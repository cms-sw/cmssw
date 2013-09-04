#ifndef SiPhase2Digitizer_h
#define SiPhase2Digitizer_h

/** \class SiPhase2Digitizer
 *
 * SiPhase2Digitizer produces digis from SimHits
 * The real algorithm is in SiPhase2DigitizerAlgorithm
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
class PixelGeomDetUnit;
class PSimHit;
class SiPhase2DigitizerAlgorithm;
class TrackerGeometry;

namespace cms {
  class SiPhase2Digitizer : public DigiAccumulatorMixMod {
  public:

    explicit SiPhase2Digitizer(const edm::ParameterSet& conf, edm::EDProducer& mixMod);

    virtual ~SiPhase2Digitizer();

    virtual void initializeEvent(edm::Event const& e, edm::EventSetup const& c) override;
    virtual void accumulate(edm::Event const& e, edm::EventSetup const& c) override;
    virtual void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c) override;
    virtual void finalizeEvent(edm::Event& e, edm::EventSetup const& c) override;

    virtual void beginJob() {}
  private:
    void accumulatePixelHits(edm::Handle<std::vector<PSimHit> >);   
    bool first;
    std::unique_ptr<SiPhase2DigitizerAlgorithm>  _pixeldigialgo;
    typedef std::vector<std::string> vstring;
    const std::string hitsProducer;
    const vstring trackerContainers;
    const std::string geometryType;
    edm::ESHandle<TrackerGeometry> pDD;
    edm::ESHandle<MagneticField> pSetup;
    std::map<unsigned int, PixelGeomDetUnit*> detectorUnits;
    CLHEP::HepRandomEngine* rndEngine;

    // infrastructure to reject dead pixels as defined in db (added by F.Blekman)
  };
}


#endif
