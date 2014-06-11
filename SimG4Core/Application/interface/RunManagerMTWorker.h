#ifndef SimG4Core_Application_RunManagerMTWorker_H
#define SimG4Core_Application_RunManagerMTWorker_H

#include "FWCore/Utilities/interface/InputTag.h"

#include "SimG4Core/Generators/interface/Generator.h"

#include <memory>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}
class Generator;

class G4Event;
class G4SimEvent;

class RunManagerMTWorker {
public:
  RunManagerMTWorker(const edm::ParameterSet& iConfig);
  ~RunManagerMTWorker();

  void produce(const edm::Event& inpevt, const edm::EventSetup& es);

private:
  G4Event *generateEvent(const edm::Event& inpevt);

  Generator m_generator;
  std::string m_InTag;

  std::unique_ptr<G4Event> m_currentEvent;
  std::unique_ptr<G4SimEvent> m_simEvent;
};

#endif
