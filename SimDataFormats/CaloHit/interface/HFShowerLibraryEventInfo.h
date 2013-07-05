#ifndef SimDataFormats_HFShowerLibraryEventInfo_h
#define SimDataFormats_HFShowerLibraryEventInfo_h

#include <vector>

class HFShowerLibraryEventInfo {

public:

  HFShowerLibraryEventInfo() { }
  HFShowerLibraryEventInfo(int events, int bins, int eventsPerBin,
			   float libraryVersion, float physListVersion, 
			   const std::vector<double> &en);
    
  // total number of events 
  int totalEvents()                const { return fEvents; }
  // number of bins 
  int numberOfBins()               const { return fBins; }
  // number of events per bin
  int eventsPerBin()               const { return fEventsPerBin; }
  // hf shower library version
  float showerLibraryVersion()     const { return fHFShowerLibVers; }
  // physics list version
  float physListVersion()          const { return fPhyListVers; }
  // energy bins
  std::vector<double> energyBins() const { return fEnergies; }
    
private:

  int                     fEvents, fBins, fEventsPerBin;
  float                   fHFShowerLibVers, fPhyListVers;
  std::vector<double>     fEnergies;
};
typedef std::vector<HFShowerLibraryEventInfo> HFShowerLibraryEventInfoCollection;
#endif
