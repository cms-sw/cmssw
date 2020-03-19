#ifndef SimG4Core_PrintTrackNumberAction_H
#define SimG4Core_PrintTrackNumberAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"

class EndOfEvent;
class EndOfTrack;

class PrintTrackNumberAction : public SimWatcher,
                               public Observer<const EndOfEvent *>,
                               public Observer<const EndOfTrack *> {
public:
  PrintTrackNumberAction(edm::ParameterSet const &p);
  ~PrintTrackNumberAction() override;
  void update(const EndOfTrack *trk) override;
  void update(const EndOfEvent *trk) override;

private:
  int theNoTracks;
  int theNoTracksThisEvent;
  int theNoTracksNoUL;
  int theNoTracksThisEventNoUL;
  int theNoTracksToPrint;
  bool bNoUserLimits;
};

#endif
