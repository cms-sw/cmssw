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
// Take these next two lines out as soon as SLHC11 samples no longer need to be processed
#include <FWCore/ServiceRegistry/interface/Service.h>
#include "RemapDetIdService.h"

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
class SiPixelDigitizerAlgorithm;
class TrackerGeometry;
class TrackerTopology;

namespace cms {
  class SiPixelDigitizer : public DigiAccumulatorMixMod {
  public:

    explicit SiPixelDigitizer(const edm::ParameterSet& conf, edm::EDProducer& mixMod);

    virtual ~SiPixelDigitizer();

    virtual void initializeEvent(edm::Event const& e, edm::EventSetup const& c) override;
    virtual void accumulate(edm::Event const& e, edm::EventSetup const& c) override;
    virtual void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c) override;
    virtual void finalizeEvent(edm::Event& e, edm::EventSetup const& c) override;

    virtual void beginJob() {}
  private:
    void accumulatePixelHits(edm::Handle<std::vector<PSimHit> >,const TrackerTopology *tTopo);   
    bool first;
    std::unique_ptr<SiPixelDigitizerAlgorithm>  _pixeldigialgo;
    typedef std::vector<std::string> vstring;
    const std::string hitsProducer;
    const vstring trackerContainers;
    const std::string geometryType;
    edm::ESHandle<TrackerGeometry> pDD;
    edm::ESHandle<MagneticField> pSetup;
    std::map<unsigned int, PixelGeomDetUnit*> detectorUnits;
    CLHEP::HepRandomEngine* rndEngine;

    // Take this member out as soon as SLHC11 files no longer need to be processed
    edm::Service<simtracker::services::RemapDetIdService> detIdRemapService_;
    // infrastructure to reject dead pixels as defined in db (added by F.Blekman)
  };
}


#endif
