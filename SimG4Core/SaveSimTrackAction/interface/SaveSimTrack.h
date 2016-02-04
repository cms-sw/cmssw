#ifndef SimG4Core_SaveSimTrack_H
#define SimG4Core_SaveSimTrack_H

#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SaveSimTrack : public SimWatcher,
                     public Observer<const BeginOfTrack *> {

public:
  SaveSimTrack(edm::ParameterSet const & p);
  ~SaveSimTrack();
  void update(const BeginOfTrack * trk);

private:
  int    pdgMin, pdgMax;
};

#endif


