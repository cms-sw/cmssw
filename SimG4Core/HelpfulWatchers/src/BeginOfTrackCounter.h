#ifndef HelpfulWatchers_BeginOfTrackCounter_h
#define HelpfulWatchers_BeginOfTrackCounter_h
// -*- C++ -*-
//
// Package:     HelpfulWatchers
// Class  :     BeginOfTrackCounter
//
/**\class BeginOfTrackCounter BeginOfTrackCounter.h
 SimG4Core/HelpfulWatchers/interface/BeginOfTrackCounter.h

 Description: Counts the number of BeginOfTrack signals and puts that value into
 the Event

 Usage:
    The module takes one optional parameter "instanceLabel" which is used to
    set the label used to put the data into the Event

*/
//
// Original Author:
//         Created:  Tue Nov 29 12:26:39 EST 2005
//

// system include files
#include <string>

// user include files
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"

// forward declarations
namespace edm {
  class ParameterSet;
}

namespace simwatcher {
  class BeginOfTrackCounter : public SimProducer, public Observer<const BeginOfTrack *> {
  public:
    BeginOfTrackCounter(const edm::ParameterSet &);
    BeginOfTrackCounter(const BeginOfTrackCounter &) = delete;                   // stop default
    const BeginOfTrackCounter &operator=(const BeginOfTrackCounter &) = delete;  // stop default

    // ---------- const member functions ---------------------

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    void produce(edm::Event &, const edm::EventSetup &) override;

  private:
    void update(const BeginOfTrack *) override;
    // ---------- member data --------------------------------
    int m_count;
    std::string m_label;
  };

}  // namespace simwatcher
#endif
