///////////////////////////////////////////////////////////////////////////////
// File: PileUpFP420.cc
// Date: 12.2006
// Description: PileUpFP420 for FP420
// Modifications:
///////////////////////////////////////////////////////////////////////////////
//#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
//#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimRomanPot/SimFP420/interface/PileUpFP420.h"
//#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

void PileUpFP420::add(const HitDigitizerFP420::hit_map_type &in, const PSimHit &hit, int verbosity) {
  if (verbosity > 0) {
    std::cout << " ==========================****PileUpFP420: add start       = " << std::endl;
  }
  for (HitDigitizerFP420::hit_map_type::const_iterator im = in.begin(); im != in.end(); im++) {
    theMap[(*im).first] += Amplitude((*im).second);

    theMapLink[(*im).first].push_back(std::pair<const PSimHit *, Amplitude>(&hit, Amplitude((*im).second)));

    if (verbosity > 0) {
      std::cout << "*********** Amplitude((*im).first)  = " << Amplitude((*im).first) << std::endl;
      std::cout << " Amplitude((*im).second)  = " << Amplitude((*im).second) << std::endl;
    }
  }  // for loop

  if (verbosity > 0) {
    std::cout << " ==========================****PileUpFP420: add end       = " << std::endl;
  }
}
void PileUpFP420::resetSignal() { theMap.clear(); }
void PileUpFP420::resetLink() { theMapLink.clear(); }
