#include "SimDataFormats/CaloHit/interface/HFShowerLibraryEventInfo.h"

HFShowerLibraryEventInfo::HFShowerLibraryEventInfo(int events, int bins, 
						   int eventsPerBin,
						   float libraryVersion,
						   float physListVersion, 
						   const std::vector<double>& en) :
  fEvents(events), fBins(bins), fEventsPerBin(eventsPerBin),
  fHFShowerLibVers(libraryVersion), fPhyListVers(physListVersion), 
  fEnergies(en) {}
