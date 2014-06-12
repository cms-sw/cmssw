#ifndef SimG4Core_Application_RunManagerMTWorker_H
#define SimG4Core_Application_RunManagerMTWorker_H

#include "FWCore/Utilities/interface/InputTag.h"

#include "SimG4Core/Generators/interface/Generator.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include <memory>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}
class Generator;
class RunManagerMT;

class G4Event;
class G4SimEvent;

class RunManagerMTWorker {
public:
  RunManagerMTWorker(const edm::ParameterSet& iConfig);
  ~RunManagerMTWorker();

  void beginRun(const RunManagerMT& runManagerMaster, const edm::EventSetup& es);

  void produce(const edm::Event& inpevt, const edm::EventSetup& es);

  void abortEvent();
  void abortRun(bool softAbort=false);

private:
  G4Event *generateEvent(const edm::Event& inpevt);

  Generator m_generator;
  std::string m_InTag;
  const bool m_nonBeam;

  std::unique_ptr<G4Event> m_currentEvent;
  std::unique_ptr<G4SimEvent> m_simEvent;

  SimActivityRegistry m_registry;
};

#endif
