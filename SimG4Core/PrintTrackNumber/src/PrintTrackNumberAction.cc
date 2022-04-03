#include "SimG4Core/PrintTrackNumber/interface/PrintTrackNumberAction.h"

#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"

#include "G4Event.hh"
#include "G4ios.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

PrintTrackNumberAction::PrintTrackNumberAction(edm::ParameterSet const &p)
    : theNoTracks(0), theNoTracksThisEvent(0), theNoTracksNoUL(0), theNoTracksThisEventNoUL(0) {
  theNoTracksToPrint = p.getUntrackedParameter<int>("EachNTrack", -1);
  // do not count tracks killed by user limits (MinEkineCut for the moment only)
  bNoUserLimits = p.getUntrackedParameter<bool>("NoUserLimits", true);
  G4cout << " PrintTrackNumberAction::bNoUserLimits " << bNoUserLimits << G4endl;
}

PrintTrackNumberAction::~PrintTrackNumberAction() {}

void PrintTrackNumberAction::update(const EndOfTrack *trk) {
  const G4Track *aTrack = (*trk)();

  theNoTracks++;
  theNoTracksThisEvent++;

  if (bNoUserLimits) {
    bool countTrk = true;
    // tracks that have been killed before first step (by MinEkineCut).
    // In fact the track makes the first step, MinEkineCut process determines
    // that the energy is too low, set it to 0, and then at the next step
    // the 0-energy particle dies
    if (aTrack->GetCurrentStepNumber() == 2) {
      const G4VProcess *proccur = aTrack->GetStep()->GetPostStepPoint()->GetProcessDefinedStep();
      if (proccur != nullptr) {
        if (proccur->GetProcessName() == "MinEkineCut") {
          countTrk = false;
        } else {
          // for e+, last step is annihil, while previous is MinEkineCut
          const G4VProcess *procprev = aTrack->GetStep()->GetPreStepPoint()->GetProcessDefinedStep();
          if (procprev != nullptr) {
            if (procprev->GetProcessName() == "MinEkineCut") {
              countTrk = false;
            }
          }
        }
      }
    }
    if (countTrk) {
      theNoTracksNoUL++;
      theNoTracksThisEventNoUL++;
      if (theNoTracksToPrint > 0) {
        if (theNoTracksThisEventNoUL % theNoTracksToPrint == 0) {
          G4cout << "PTNA: Simulating Track Number = " << theNoTracksThisEventNoUL << G4endl;
        }
      }
    }
  } else {
    if (theNoTracksToPrint > 0) {
      if (theNoTracksThisEvent % theNoTracksToPrint == 0) {
        G4cout << "PTNA: Simulating Track Number = " << theNoTracksThisEvent << G4endl;
      }
    }
  }
}

void PrintTrackNumberAction::update(const EndOfEvent *e) {
  const G4Event *g4e = (*e)();
  G4cout << "PTNA: Event simulated= " << g4e->GetEventID() << " #tracks= ";
  if (bNoUserLimits) {
    G4cout << theNoTracksThisEventNoUL << "  Total #tracks in run= " << theNoTracksNoUL
           << " counting killed by UL= " << theNoTracks << G4endl;
    theNoTracksThisEventNoUL = 0;
  } else {
    G4cout << theNoTracksThisEvent << "  Total #tracks in run= " << theNoTracks << G4endl;
    theNoTracksThisEvent = 0;
  }
}
