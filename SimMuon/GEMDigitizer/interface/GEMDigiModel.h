#ifndef SimMuon_GEMDigitizer_GEMDigiModel_h
#define SimMuon_GEMDigitizer_GEMDigiModel_h

/** 
 *  \class GEMDigiModel
 *
 *  Base Class for the GEM strip response simulation 
 *  
 *  \author Sven Dildick
 *  \modified by Yechan Kang
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

class PSimHit;
class GEMEtaPartition;
class GEMGeometry;

typedef std::set<std::pair<int, int> > Strips;
typedef std::multimap<std::pair<unsigned int, int>, const PSimHit *, std::less<std::pair<unsigned int, int> > >
    DetectorHitMap;

class GEMDigiModel {
public:
  virtual ~GEMDigiModel() {}

  virtual void simulate(
      const GEMEtaPartition *, const edm::PSimHitContainer &, CLHEP::HepRandomEngine *, Strips &, DetectorHitMap &) = 0;

  void setGeometry(const GEMGeometry *geom) { geometry_ = geom; }

protected:
  const GEMGeometry *geometry_;
  GEMDigiModel(const edm::ParameterSet &) {}
};
#endif
