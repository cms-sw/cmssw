#ifndef SimG4Core_DBremWatcher_H
#define SimG4Core_DBremWatcher_H

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "G4ThreeVector.hh"

#include <vector>
#include <tuple>

class DBremWatcher : public SimProducer,
                     public Observer<const BeginOfTrack *>,
                     public Observer<const BeginOfEvent *>,
                     public Observer<const BeginOfRun *>,
                     public Observer<const EndOfEvent *>,
                     public Observer<const EndOfTrack *> {
public:
  DBremWatcher(edm::ParameterSet const &p);
  ~DBremWatcher() override;
  void update(const BeginOfTrack *trk) override;
  void update(const BeginOfEvent *event) override;
  void update(const EndOfEvent *event) override;
  void update(const BeginOfRun *run) override;
  void update(const EndOfTrack *trk) override;
  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  std::vector<int> pdgs_;
  int MotherId;
  float m_weight;
  double biasFactor;
  bool foundbrem;
  G4ThreeVector aPrimeTraj;
  G4ThreeVector finaltraj;
  G4ThreeVector VertexPos;
  float f_energy;
};

#endif
