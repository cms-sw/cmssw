#ifndef SimGeneral_MixingModule_DigiAccumulatorMixMod_h
#define SimGeneral_MixingModule_DigiAccumulatorMixMod_h
// -*- C++ -*-
//
// Package:     MixingModule
// Class  :     DigiAccumulatorMixMod
//
/**\class DigiAccumulatorMixMod DigiAccumulatorMixMod.h SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Tue Sep 20 17:33:33 CEST 2011
//

// system include files
#include <vector>

// user include files

// forward declarations

namespace edm {
  class Event;
  class EventPrincipal;
  class EventSetup;
  class LuminosityBlock;
  class Run;
}

class PileUpEventPrincipal;

class DigiAccumulatorMixMod {

  public:
    DigiAccumulatorMixMod();

    virtual ~DigiAccumulatorMixMod();

    // ---------- const member functions ---------------------

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    // Perform algorithm setup for each event before any signal/pileup events are processed.
    virtual void initializeEvent(edm::Event const& event, edm::EventSetup const& setup) = 0;

    // Accumulate digis or other data for each signal event.
    virtual void accumulate(edm::Event const& event, edm::EventSetup const& setup) = 0;

    // Accumulate digis or other data for each pileup event, one at a time.
    virtual void accumulate(PileUpEventPrincipal const& event, edm::EventSetup const& setup) = 0;

    // 1. Finalize digi collections or other data for each event.
    // 2. Put products in Event with appropriate instance labels
    virtual void finalizeEvent(edm::Event& event, edm::EventSetup const& setup) = 0;  // event is non-const

    // Perform any necessary actions before pileup events for a given bunch crossing are processed.
    virtual void initializeBunchCrossing(edm::Event const& event, edm::EventSetup const& setup, int bunchCrossing) {}

    // Perform any necessary actions after pileup events for a given bunch crossing are processed.
    // This may include putting bunch crossing specific products into the event.
    virtual void finalizeBunchCrossing(edm::Event& event, edm::EventSetup const& setup, int bunchCrossing) {}

    virtual void beginRun(edm::Run const& run, edm::EventSetup const& setup) {}
    virtual void endRun(edm::Run const& run, edm::EventSetup const& setup) {}
    virtual void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {}
    virtual void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {}

  private:
    DigiAccumulatorMixMod(DigiAccumulatorMixMod const&); // stop default

    DigiAccumulatorMixMod const& operator=(DigiAccumulatorMixMod const&); // stop default

    // ---------- member data --------------------------------
};

#endif
