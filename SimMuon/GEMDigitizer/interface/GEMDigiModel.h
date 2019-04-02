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
#include "SimMuon/GEMDigitizer/interface/GEMDigiModule.h"

#include <map>
#include <set>

namespace CLHEP {
  class HepRandomEngine;
}

class PSimHit;
class GEMEtaPartition;
class GEMDigiModule;

class GEMDigiModel
{
public:

  virtual ~GEMDigiModel() {}

  virtual void simulate(const GEMEtaPartition*, const edm::PSimHitContainer&, CLHEP::HepRandomEngine*) = 0;

protected:

  GEMDigiModule* digiModule_;

  GEMDigiModel(const edm::ParameterSet&, GEMDigiModule* digiModule) : digiModule_(digiModule) {}

};
#endif
