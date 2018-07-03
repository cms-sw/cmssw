#ifndef SimMuon_GEMDigitizer_GEMDigiModel_h
#define SimMuon_GEMDigitizer_GEMDigiModel_h

/** 
 *  \class GEMDigiModel
 *
 *  Base Class for the GEM strip response simulation 
 *  
 *  \author Sven Dildick
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/GEMDigiSimLink/interface/GEMDigiSimLink.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include <map>
#include <set>

namespace CLHEP {
  class HepRandomEngine;
}

class GEMEtaPartition;
class GEMGeometry;
class PSimHit;

class GEMDigiModel
{
public:

  typedef edm::DetSet<StripDigiSimLink> StripDigiSimLinks;
  typedef edm::DetSet<GEMDigiSimLink> GEMDigiSimLinks;

  virtual ~GEMDigiModel() {}

  void setGeometry(const GEMGeometry *geom) {geometry_ = geom;}

  const GEMGeometry* getGeometry() {return geometry_;}

  virtual void simulateSignal(const GEMEtaPartition*, const edm::PSimHitContainer&, CLHEP::HepRandomEngine* engine) = 0;

  virtual void simulateNoise(const GEMEtaPartition*, CLHEP::HepRandomEngine* engine) = 0;
  
  virtual std::vector<std::pair<int,int> > 
    simulateClustering(const GEMEtaPartition*, const PSimHit*, const int, CLHEP::HepRandomEngine* engine) = 0;

  void fillDigis(int rollDetId, GEMDigiCollection&);

  virtual void setup() = 0;

  const StripDigiSimLinks & stripDigiSimLinks() const {return stripDigiSimLinks_;}
  const GEMDigiSimLinks & gemDigiSimLinks() const {return theGemDigiSimLinks_;}

protected:

  GEMDigiModel(const edm::ParameterSet&) {}

  const GEMGeometry * geometry_;
  
  std::set< std::pair<int, int> > strips_;

  /// creates links from Digi to SimTrack
  void addLinks(unsigned int strip,int bx);
  void addLinksWithPartId(unsigned int strip,int bx);

  // keeps track of which hits contribute to which channels
  typedef std::multimap<
      std::pair<unsigned int, int>,
      const PSimHit*,
      std::less<std::pair<unsigned int, int> >
    >  DetectorHitMap;

  DetectorHitMap detectorHitMap_;
  StripDigiSimLinks stripDigiSimLinks_;
  GEMDigiSimLinks theGemDigiSimLinks_;

};
#endif
