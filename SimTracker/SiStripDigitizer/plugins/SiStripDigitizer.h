/** \class SiStripDigitizer
 *
 * SiStripDigitizer to convert hits to digis
 *
 ************************************************************/

#ifndef SiStripDigitizer_h
#define SiStripDigitizer_h

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "FWCore/Framework/interface/ESHandle.h"

class TrackerTopology;

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
class SiStripDigitizerAlgorithm;
class StripGeomDetUnit;
class TrackerGeometry;

class SiStripDigitizer : public DigiAccumulatorMixMod {
public:
  explicit SiStripDigitizer(const edm::ParameterSet& conf, edm::EDProducer& mixMod);
  
  virtual ~SiStripDigitizer();
  
  virtual void initializeEvent(edm::Event const& e, edm::EventSetup const& c);
  virtual void accumulate(edm::Event const& e, edm::EventSetup const& c);
  virtual void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c);
  virtual void finalizeEvent(edm::Event& e, edm::EventSetup const& c);
  
private:
  void accumulateStripHits(edm::Handle<std::vector<PSimHit> >, const TrackerTopology *tTopo);   

  typedef std::vector<std::string> vstring;
  typedef std::map<unsigned int, std::vector<std::pair<const PSimHit*, int> >,std::less<unsigned int> > simhit_map;
  typedef simhit_map::iterator simhit_map_iterator;

  const std::string gainLabel;
  const std::string hitsProducer;
  const vstring trackerContainers;
  const std::string ZSDigi;
  const std::string SCDigi;
  const std::string VRDigi;
  const std::string PRDigi;
  const std::string geometryType;
  const bool useConfFromDB;
  const bool zeroSuppression;

  std::unique_ptr<SiStripDigitizerAlgorithm> theDigiAlgo;
  std::map<uint32_t, std::vector<int> > theDetIdList;
  edm::ESHandle<TrackerGeometry> pDD;
  edm::ESHandle<MagneticField> pSetup;
  std::map<unsigned int, StripGeomDetUnit*> detectorUnits;

  CLHEP::HepRandomEngine* rndEngine;
};

#endif
