///////////////////////////////////////////////////////////////////////////////
// File: PileUpFP420.cc
// Date: 12.2006
// Description: PileUpFP420 for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
//#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
//#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
#include "SimRomanPot/SimFP420/interface/PileUpFP420.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
//#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
//#define mydigidebug6

void PileUpFP420::add(HitDigitizerFP420::hit_map_type in, const PSimHit& hit){


#ifdef mydigidebug6
	  std::cout << " ==========================****PileUpFP420: add start       = " << std::endl;
#endif
       for (HitDigitizerFP420::hit_map_type::const_iterator im = in.begin(); im!=in.end(); im++ ){

	  theMap[(*im).first] += Amplitude((*im).second);

	  theMapLink[(*im).first].push_back( pair < const PSimHit*, Amplitude >                                                     (  &hit,  Amplitude((*im).second)   )              );

#ifdef mydigidebug6
	  std::cout << "*********** Amplitude((*im).first)  = " << Amplitude((*im).first)  << std::endl;
	  std::cout << " Amplitude((*im).second)  = " << Amplitude((*im).second)  << std::endl;
#endif
	} // for loop
#ifdef mydigidebug6
	  std::cout << " ==========================****PileUpFP420: add end       = " << std::endl;
#endif
}
void PileUpFP420::resetSignal(){
  theMap.clear();
}
void PileUpFP420::resetLink(){
  theMapLink.clear();
}
