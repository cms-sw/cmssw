#include "SimG4CMS/Tracker/interface/TkSimTrackSelection.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "G4Track.hh"

#include <iostream>

//#define DEBUG
//#define DUMPPROCESSES

#ifdef DUMPPROCESSES
#include "G4VProcess.hh"
#endif

using std::cout;
using std::endl;

TkSimTrackSelection::TkSimTrackSelection() : rTracker(1200*mm), zTracker(3000*mm) 
{
//   energyCut = SimpleConfigurable<float>(0.5,"TkSimTrackSelection:EnergyThresholdForPersistencyInGeV")*GeV;
//   energyHistoryCut = SimpleConfigurable<float>(0.05,"TkSimTrackSelection:EnergyThresholdForHistoryInGeV")*GeV;
    energyCut = 0.5*GeV;
    energyHistoryCut = 0.05*GeV;
    cout << " Criteria for Saving Tracker SimTracks:  ";
    cout << " History: " << energyHistoryCut << " MeV ; Persistency: " << energyCut <<"  MeV " << endl;
} 

void TkSimTrackSelection::upDate(const BeginOfTrack * bot)
{
    const G4Track* gTrack = (*bot)();
#ifdef DUMPPROCESSES
    cout << " -> process creator pointer " << gTrack->GetCreatorProcess() << endl;
    if (gTrack->GetCreatorProcess())
	cout << " -> PROCESS CREATOR : " << gTrack->GetCreatorProcess()->GetProcessName() << endl;
#endif
    //Position
    const G4ThreeVector pos = gTrack->GetPosition();
#ifdef DEBUG
    cout << " ENERGY MeV " << gTrack->GetKineticEnergy() << " Energy Cut" << energyCut << endl;
    cout << " TOTAL ENERGY " << gTrack->GetTotalEnergy() << endl;
    cout << " WEIGHT " << gTrack->GetWeight() << endl;
#endif
    // Check if in Tracker Volume
    //
    if (pos.perp() < rTracker && std::abs(pos.z()) < zTracker)
    {
	// inside the Tracker
	//
#ifdef DEBUG
	cout << " INSIDE TRACKER" << endl;
#endif
	if (gTrack->GetKineticEnergy() > energyCut)
	{
	    TrackInformation * info = getOrCreateTrackInformation(gTrack);
#ifdef DEBUG
	    cout << " POINTER " << info << endl;
	    cout << " track inside the tracker selected for STORE" << endl;
#endif
	    info->storeTrack(true);
	}
	// Save History?
	//
	if (gTrack->GetKineticEnergy() > energyHistoryCut)
	{
	    TrackInformation * info = getOrCreateTrackInformation(gTrack);
	    info->putInHistory();
#ifdef DEBUG
	    cout << " POINTER " << info << endl;
	    cout << " track inside the tracker selected for HISTORY" << endl;
#endif
	}
    }
}

TrackInformation * TkSimTrackSelection::getOrCreateTrackInformation(const G4Track * gTrack)
{
    G4VUserTrackInformation* temp = gTrack->GetUserInformation();
    if (temp == 0)
    {
	cout << " ERROR: no G4VUserTrackInformation available" << endl;
	abort();
    }
    else
    {
	TrackInformation * info = dynamic_cast<TrackInformation*>(temp);
	if (info ==0)
	{
	    cout << " ERROR: TkSimTrackSelection: the UserInformation does not appear to be a TrackInformation" << endl;
	    abort();
	}
	return info;
    }
}
