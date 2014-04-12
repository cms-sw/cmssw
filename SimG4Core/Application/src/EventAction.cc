#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Application/interface/SimRunInterface.h"
#include "SimG4Core/Application/interface/G4SimVertex.h"
#include "SimG4Core/Application/interface/G4SimTrack.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <fstream>

//using std::cout;
//using std::endl;

EventAction::EventAction(const edm::ParameterSet & p,
                         SimRunInterface* rm,
			 SimTrackManager* iManager) 
    : m_runInterface(rm),
      m_trackManager(iManager),
      m_stopFile(p.getParameter<std::string>("StopFile")),
      m_debug(p.getUntrackedParameter<bool>("debug",false))
{
  m_trackManager->setCollapsePrimaryVertices(p.getParameter<bool>("CollapsePrimaryVertices"));
}

EventAction::~EventAction() {}
    
void EventAction::BeginOfEventAction(const G4Event * anEvent)
{
  if (std::ifstream(m_stopFile.c_str()))
    {
      edm::LogWarning("SimG4CoreApplication")
        << "BeginOfEventAction: termination signal received at event "
	<< anEvent->GetEventID();
      /*
        cout << "BeginOfEventAction: termination signal received at event "
             << anEvent->GetEventID() << endl;
      */
      m_runInterface->abortRun(true);
    }

    m_trackManager->reset();
    BeginOfEvent e(anEvent);
    m_beginOfEventSignal(&e);

}

void EventAction::EndOfEventAction(const G4Event * anEvent)
{
  if (std::ifstream(m_stopFile.c_str()))
    {
      edm::LogWarning("SimG4CoreApplication")
        << "EndOfEventAction: termination signal received at event "
	<< anEvent->GetEventID();
      /*
        cout << "EndOfEventAction: termination signal received at event "
             << anEvent->GetEventID() << endl;
      */
      // soft abort run
      m_runInterface->abortRun(true);
    }
  if (anEvent->GetNumberOfPrimaryVertex()==0)
    {
      edm::LogWarning("SimG4CoreApplication")
        << "EndOfEventAction: event " << anEvent->GetEventID() 
	<< " must have failed (no G4PrimaryVertices found) and will be skipped ";
      /*
        cout << " EndOfEventAction: event " << anEvent->GetEventID()
             << " must have failed (no G4PrimaryVertices found) and will be skipped " 
	     << endl;
      */
      return;
    }

  m_trackManager->storeTracks(m_runInterface->simEvent());

  // dispatch now end of event, and only then delete tracks...
  EndOfEvent e(anEvent);
  m_endOfEventSignal(&e);

  m_trackManager->deleteTracks();
  m_trackManager->cleanTkCaloStateInfoMap();
}

void EventAction::addTrack(TrackWithHistory* iTrack, bool inHistory, bool withAncestor)
{
  m_trackManager->addTrack(iTrack, inHistory, withAncestor);
}

void EventAction::addTkCaloStateInfo(uint32_t t,const std::pair< math::XYZVectorD,
				     math::XYZTLorentzVectorD>& p)
{
  m_trackManager->addTkCaloStateInfo(t,p);
}

void EventAction::abortEvent()
{
  m_runInterface->abortEvent();
}
