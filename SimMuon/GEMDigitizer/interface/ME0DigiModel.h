#ifndef ME0Digitizer_ME0DigiModel_h
#define ME0Digitizer_ME0DigiModel_h

/** 
 *  \class ME0DigiModel
 *
 *  Base Class for the ME0 strip response simulation 
 *  
 *  \author Sven Dildick
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GEMDigi/interface/ME0DigiCollection.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "CLHEP/Random/RandomEngine.h"

#include <map>
#include <set>

class ME0EtaPartition;
class ME0Geometry;
class PSimHit;

class ME0DigiModel
{
public:

  typedef edm::DetSet<StripDigiSimLink> StripDigiSimLinks;

  virtual ~ME0DigiModel() {}

  void setGeometry(const ME0Geometry *geom) {geometry_ = geom;}

  const ME0Geometry* getGeometry() {return geometry_;}

  virtual void simulateSignal(const ME0EtaPartition*, const edm::PSimHitContainer&) = 0;

  virtual void simulateNoise(const ME0EtaPartition*) = 0;
  
  virtual std::vector<std::pair<int,int> > 
    simulateClustering(const ME0EtaPartition*, const PSimHit*, const int) = 0;

  virtual void setRandomEngine(CLHEP::HepRandomEngine&) = 0;

  void fillDigis(int rollDetId, ME0DigiCollection&);

  virtual void setup() = 0;

  const StripDigiSimLinks & stripDigiSimLinks() const {return stripDigiSimLinks_;}

protected:

  ME0DigiModel(const edm::ParameterSet&) {}

  const ME0Geometry * geometry_;
  
  std::set< std::pair<int, int> > strips_;

  /// creates links from Digi to SimTrack
  void addLinks(unsigned int strip,int bx);

  // keeps track of which hits contribute to which channels
  typedef std::multimap<
      std::pair<unsigned int, int>,
      const PSimHit*,
      std::less<std::pair<unsigned int, int> >
    >  DetectorHitMap;

  DetectorHitMap detectorHitMap_;
  StripDigiSimLinks stripDigiSimLinks_;
};
#endif
