#ifndef GEMDigitizer_GEMDigiModel_h
#define GEMDigitizer_GEMDigiModel_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "CLHEP/Random/RandomEngine.h"

#include "SimMuon/GEMDigitizer/src/GEMTiming.h"
#include "SimMuon/GEMDigitizer/src/GEMNoise.h"
#include "SimMuon/GEMDigitizer/src/GEMClustering.h"
#include "SimMuon/GEMDigitizer/src/GEMEfficiency.h"

#include <map>
#include <set>

class GEMEtaPartition;
class PSimHit;
class GEMGeometry;
class GEMStripNoise; 
class GEMStripNoiseRcd; 
class GEMStripClustering; 
class GEMStripClusteringRcd; 
class GEMStripEfficiency; 
class GEMStripEfficiencyRcd; 
class GEMStripTiming; 
class GEMStripTimingRcd; 


namespace CLHEP 
{ 
  class HepRandomEngine; 
} 


class GEMDigiModel
{
 public:

  typedef edm::DetSet<StripDigiSimLink> StripDigiSimLinks;

  GEMDigiModel(const edm::ParameterSet&);
  ~GEMDigiModel();

  const StripDigiSimLinks & stripDigiSimLinks() const {return stripDigiSimLinks_;}
  
  void setGeometry(const GEMGeometry*);

  const GEMGeometry * getGeometry() const {return geometry_;}

  void setRandomEngine(CLHEP::HepRandomEngine& eng);

  void setUp(std::vector<GEMStripTiming::StripTimingItem>, 
	     std::vector<GEMStripNoise::StripNoiseItem>,  
	     std::vector<GEMStripClustering::StripClusteringItem>, 
	     std::vector<GEMStripEfficiency::StripEfficiencyItem>);  

  void simulateSignal(const GEMEtaPartition*, const edm::PSimHitContainer&);

  void simulateNoise(const GEMEtaPartition*);
  
  void fillDigis(const uint32_t, GEMDigiCollection&);

 private:

  void addLinks(int, int);

  const GEMGeometry* geometry_;

  std::string digiModelString_;

  GEMTiming* timingModel_ = 0;
  GEMNoise* noiseModel_ = 0; 
  GEMClustering* clusteringModel_ = 0; 
  GEMEfficiency* efficiencyModel_ = 0; 

  std::set< std::pair<int, int> > strips_;

  // keeps track of which hits contribute to which channels
  // isn't the multimap a bit too complicated?
  typedef std::multimap<
    std::pair<unsigned int, int>,
    const PSimHit*,
    std::less<std::pair<unsigned int, int> >
    >  DetectorHitMap;

  DetectorHitMap detectorHitMap_;
  StripDigiSimLinks stripDigiSimLinks_;
};

#endif
