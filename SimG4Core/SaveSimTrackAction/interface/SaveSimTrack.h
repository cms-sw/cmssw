#ifndef SimG4Core_SaveSimTrack_H
#define SimG4Core_SaveSimTrack_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include <vector>

class SaveSimTrack : public SimWatcher, public Observer<const BeginOfTrack *> {
public:
  SaveSimTrack(edm::ParameterSet const &p);
  ~SaveSimTrack() override;
  void update(const BeginOfTrack *trk) override;

private:
  std::vector<int> pdgs_;
};

#endif
