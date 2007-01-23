#ifndef SimG4Core_EventAction_H
#define SimG4Core_EventAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Application/interface/G4SimEvent.h"
#include "SimG4Core/Application/interface/SimTrackManager.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/TrackContainer.h" 
#include "SimG4Core/Notification/interface/SimActivityRegistry.h" 

#include "G4UserEventAction.hh"

#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Vector/ThreeVector.h"

#include <vector>
#include <map>
 
class RunManager;
class BeginOfEvent;
class EndOfEvent;
 
class EventAction: public G4UserEventAction
{
public:
    //EventAction(const edm::ParameterSet & ps);
    EventAction(const edm::ParameterSet & ps,
		SimTrackManager*);
    ~EventAction();
    void BeginOfEventAction(const G4Event * evt);
    void EndOfEventAction(const G4Event * evt);

    const TrackContainer * trackContainer() const { 
      return m_trackManager->trackContainer();
    }
    void addTrack(TrackWithHistory* iTrack);
    void addTkCaloStateInfo(uint32_t t,std::pair<Hep3Vector,HepLorentzVector> p); 

    SimActivityRegistry::BeginOfEventSignal m_beginOfEventSignal;
    SimActivityRegistry::EndOfEventSignal m_endOfEventSignal;

private:
    //does not own the manager
    SimTrackManager* m_trackManager;
    std::string m_stopFile;
    bool m_debug;
};

#endif



