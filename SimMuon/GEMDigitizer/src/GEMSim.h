#ifndef GEMDigitizer_GEMSim_h
#define GEMDigitizer_GEMSim_h

/** \class GEMSim
 *   Base Class for the GEM strip response simulation
 *  
 *  \author Vadim Khotilovich
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "CLHEP/Random/RandomEngine.h"

#include <map>
#include <set>

class GEMEtaPartition;
class GEMGeometry;
class GEMSimSetUp;
class PSimHit;

class GEMSim
{
public:

  typedef edm::DetSet<StripDigiSimLink> StripDigiSimLinks;

  virtual ~GEMSim() {}

  virtual void simulate(const GEMEtaPartition* roll, const edm::PSimHitContainer& rpcHits) = 0;

  virtual void simulateNoise(const GEMEtaPartition* roll) = 0;

  virtual void setRandomEngine(CLHEP::HepRandomEngine& eng) = 0;

  virtual void fillDigis(int rollDetId, GEMDigiCollection& digis);

  void setGEMSimSetUp(GEMSimSetUp* setup) { simSetUp_ = setup; }

  GEMSimSetUp* getGEMSimSetUp() { return simSetUp_; }

  const StripDigiSimLinks & stripDigiSimLinks() const { return stripDigiSimLinks_; }

protected:

  GEMSim(const edm::ParameterSet& config) {}

  virtual void init() = 0;

  std::set< std::pair<int, int> > strips_;

  /// creates links from Digi to SimTrack
  /// disabled for now
  virtual void addLinks(unsigned int strip,int bx);

  // keeps track of which hits contribute to which channels
  typedef std::multimap<
      std::pair<unsigned int, int>,
      const PSimHit*,
      std::less<std::pair<unsigned int, int> >
    >  DetectorHitMap;

  DetectorHitMap detectorHitMap_;
  StripDigiSimLinks stripDigiSimLinks_;

  GEMSimSetUp* simSetUp_;
};
#endif
