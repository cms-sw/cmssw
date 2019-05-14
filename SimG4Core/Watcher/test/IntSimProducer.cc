// -*- C++ -*-
//
// Package:     Watcher
// Class  :     IntSimProducer
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Tue Nov 29 11:33:37 EST 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

class IntSimProducer : public SimProducer {
public:
  IntSimProducer(const edm::ParameterSet &);

  void produce(edm::Event &e, const edm::EventSetup &) override {
    std::unique_ptr<int> newInt(new int(++m_int));
    e.put(std::move(newInt));
  }

private:
  int m_int;
};
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
IntSimProducer::IntSimProducer(const edm::ParameterSet &) {
  m_int = 0;
  produces<int>();
}

DEFINE_SIMWATCHER(IntSimProducer);
