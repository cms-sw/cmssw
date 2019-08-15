#ifndef SimMuon_GEMDigitizer_GEMDigiModule_h
#define SimMuon_GEMDigitizer_GEMDigiModule_h

/** 
 *  \class GEMDigiModule
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
#include "SimMuon/GEMDigitizer/interface/GEMDigiModel.h"

#include <map>
#include <set>

namespace CLHEP {
  class HepRandomEngine;
}

class GEMDigiModel;
class GEMEtaPartition;
class GEMGeometry;
class PSimHit;

class GEMDigiModule {
public:
  GEMDigiModule(const edm::ParameterSet&);

  ~GEMDigiModule();

  typedef edm::DetSet<StripDigiSimLink> StripDigiSimLinks;
  typedef edm::DetSet<GEMDigiSimLink> GEMDigiSimLinks;

  void setGeometry(const GEMGeometry*);

  void simulate(const GEMEtaPartition*, const edm::PSimHitContainer&, CLHEP::HepRandomEngine*);

  void fillDigis(int rollDetId, GEMDigiCollection&);

  const StripDigiSimLinks& stripDigiSimLinks() const { return stripDigiSimLinks_; }
  const GEMDigiSimLinks& gemDigiSimLinks() const { return theGemDigiSimLinks_; }

private:
  const GEMGeometry* geometry_;

  std::vector<std::unique_ptr<GEMDigiModel> > models;

  /// creates links from Digi to SimTrack
  void addLinks(unsigned int strip, int bx);
  void addLinksWithPartId(unsigned int strip, int bx);

  Strips strips_;
  DetectorHitMap detectorHitMap_;
  StripDigiSimLinks stripDigiSimLinks_;
  GEMDigiSimLinks theGemDigiSimLinks_;
};
#endif
