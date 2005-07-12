#include "SimG4Core/Application/interface/RunManager.h"
#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Application/interface/G4SimTrack.h"
#include "SimG4Core/Application/interface/G4SimVertex.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

#include <fstream>

using std::cout;
using std::endl;

EventAction::EventAction(const edm::ParameterSet & p) 
    : m_trksForThisEvent(0),m_nVertices(0),
      m_collapsePrimaryVertices(p.getParameter<bool>("CollapsePrimaryVertices")),
      m_stopFile(p.getParameter<std::string>("StopFile")),m_debug(p.getParameter<bool>("debug"))
{}

EventAction::~EventAction() {}
    
void EventAction::BeginOfEventAction(const G4Event * anEvent)
{
    if (std::ifstream(m_stopFile.c_str()))
    {
        cout << "BeginOfEventAction: termination signal received at event "
             << anEvent->GetEventID() << endl;
        RunManager::instance()->abortRun(true);
    }
    if (m_trksForThisEvent==0) m_trksForThisEvent = new TrackContainer();
    else
    {
	for (unsigned int i = 0; i < m_trksForThisEvent->size(); i++) 
	    delete (*m_trksForThisEvent)[i];
	delete m_trksForThisEvent;
	m_trksForThisEvent = new TrackContainer();
    }
    BeginOfEvent e(anEvent);
    cleanVertexMap();
}

void EventAction::EndOfEventAction(const G4Event * anEvent)
{
    if (std::ifstream(m_stopFile.c_str()))
    {
        cout << "EndOfEventAction: termination signal received at event "
             << anEvent->GetEventID() << endl;
        RunManager::instance()->abortRun(true);
    }
    if (anEvent->GetNumberOfPrimaryVertex()==0)
    {
        cout << " EndOfEventAction: event " << anEvent->GetEventID()
             << " must have failed (no G4PrimaryVertices found) and will be skipped " << endl;
        return;
    }
#ifdef DEBUG
    cout << " EndOfEventAction knows " << m_trksForThisEvent->size()
	 << " tracks with history before branching" <<endl;
    for (unsigned int it =0;  it <(*m_trksForThisEvent).size(); it++)
	cout << " 1 - Track in position " << it << " G4 track number "
	     << (*m_trksForThisEvent)[it]->trackID()
	     << " mother " << (*m_trksForThisEvent)[it]->parentID()
	     << " status " << (*m_trksForThisEvent)[it]->saved() << endl;
#endif
    for (unsigned int i = 0; i < m_trksForThisEvent->size(); i++)
    {
	TrackWithHistory * t = (*m_trksForThisEvent)[i];
	if (t->saved()) saveTrackAndItsBranch(i);
    }
#ifdef DEBUG
    cout << "EventAction -  TRACKS to be saved starting with "
	 << (*m_trksForThisEvent).size() << endl;
    for (unsigned int it = 0;  it < (*m_trksForThisEvent).size(); it++)
	cout << " 2 - Track in position " << it
	     << " G4 track number " << (*m_trksForThisEvent)[it]->trackID()
	     << " mother " << (*m_trksForThisEvent)[it]->parentID()
	     << " Status " << (*m_trksForThisEvent)[it]->saved() << endl;
#endif
    // now eliminate from the vector the tracks with only history but not save
    int num = 0;
    for (unsigned int it = 0;  it < (*m_trksForThisEvent).size(); it++)
    {
	if ((*m_trksForThisEvent)[it]->saved() == true)
	{
	    if (it>num) (*m_trksForThisEvent)[num] = (*m_trksForThisEvent)[it];
	    num++;
	}
	else delete (*m_trksForThisEvent)[it];
    }
    (*m_trksForThisEvent).resize(num);
#ifdef DEBUG
    cout << " AFTER CLEANING, I GET " << (*m_trksForThisEvent).size()
	 << " tracks to be saved persistently" << endl;
    cout << "EventAction -  Tracks still alive " << (*m_trksForThisEvent).size() << endl;
    for (unsigned int it = 0;  it < (*m_trksForThisEvent).size(); it++)
	cout << " 3 - Track in position " << it
	     << " G4 track number " << (*m_trksForThisEvent)[it]->trackID()
	     << " mother " << (*m_trksForThisEvent)[it]->parentID()
	     << " status " << (*m_trksForThisEvent)[it]->saved() << endl;
#endif
    stable_sort(m_trksForThisEvent->begin(),m_trksForThisEvent->end(),trkIDLess());
    // reorder from 0 to max-1
    m_g4ToSimMap.clear();
    m_simToG4Vector.clear();
    for (unsigned int i = 0; i < m_trksForThisEvent->size(); i++)
    {
#ifdef DEBUG
	cout << " g4ToSimMap filling "
	     << ((*m_trksForThisEvent)[i])->trackID() << " " << i
	     << " mother " << ((*m_trksForThisEvent)[i])->parentID() << endl;
#endif
	int g4ID = (*m_trksForThisEvent)[i]->trackID();
	m_g4ToSimMap[g4ID] = i;
	m_simToG4Vector.push_back(g4ID);
	((*m_trksForThisEvent)[i])->setTrackID(i);
    }
    // second iteration : change also the parent id to the new schema
    for (unsigned int i = 0; i < m_trksForThisEvent->size(); i++)
    {
	int oldParentId = ((*m_trksForThisEvent)[i])->parentID();
	((*m_trksForThisEvent)[i])->setParentID(g4ToSim(oldParentId));
    }
    // really store tracks
    storeTracks(RunManager::instance()->simEvent());
    // dispatch now end of event, and only then delete tracks...
    EndOfEvent e(anEvent);

    for (unsigned int i = 0; i < m_trksForThisEvent->size(); i++) delete (*m_trksForThisEvent)[i];
    delete m_trksForThisEvent;
    m_trksForThisEvent = 0;
}

/// this saves a track and all its parents looping over the non ordered vector
void EventAction::saveTrackAndItsBranch(int i)
{
    TrackWithHistory * trkH = (*m_trksForThisEvent)[i];
    if (trkH == 0)
    {
        cout << " EventAction::saveTrackAndItsBranch got 0 pointer " << endl;
        abort();
    }
    trkH->save();
    unsigned int parent = trkH->parentID();
    bool parentExists=false;
    int numParent=-1;
    // search for parent. please note that now the vector is not ordered nor compact
    for (unsigned int it = 0; it < m_trksForThisEvent->size(); it++)
    {
        if ((*m_trksForThisEvent)[it]->trackID() == parent)
        {
            numParent = it;
            parentExists=true;
            break;
        }
    }
    if (parentExists) saveTrackAndItsBranch(numParent);
}

void EventAction::storeTracks(G4SimEvent * simEvent)
{
    // loop over the (now ordered) vector and really save the tracks
    for (unsigned int it = 0; it < m_trksForThisEvent->size(); it++)
    {
        TrackWithHistory * trkH = (*m_trksForThisEvent)[it];
        // at this stage there is one vertex per track, so the vertex id of track N is also N
        int ivertex = -1;
        int ig;
        Hep3Vector pm = 0.;
        int iParentID = trkH->parentID();
        ig = trkH->genParticleID();
        if (iParentID != InvalidID) pm = (*m_trksForThisEvent)[iParentID]->momentum();
        ivertex = getOrCreateVertex(trkH,iParentID,simEvent);
	simEvent->add(new G4SimTrack(trkH->trackID(),trkH->particleID(),
                                    trkH->momentum(),trkH->totalEnergy(),ivertex,ig,pm));
    }
}

int EventAction::getOrCreateVertex(TrackWithHistory * trkH, int iParentID,
				   G4SimEvent * simEvent)
{
    // if iParentID is invalid, always create the vertex
    if (iParentID == InvalidID  && m_collapsePrimaryVertices == false)
    {
	simEvent->add(new G4SimVertex(trkH->vertexPosition(),trkH->globalTime(),-1));
	m_nVertices++;
	return (m_nVertices-1);
    }
    else
    {
	// if is valid, search if it has already been saved and return the number
	VertexMap::const_iterator iterator = m_vertexMap.find(iParentID);
	if (iterator != m_vertexMap.end())
        {
	    // loop over saved vertices
	    for (unsigned int k=0; k<m_vertexMap[iParentID].size(); k++)
            {
		if ((trkH->vertexPosition()-(((m_vertexMap[iParentID])[k]).second)).mag()<0.001)
		    return (((m_vertexMap[iParentID])[k]).first);
            }
        }
	int realParent = iParentID;
	if (iParentID == InvalidID) 
	    // needed to collapse primary vertices; save a vertex with no decaying particle
	    realParent = -1;
	simEvent->add(new G4SimVertex(trkH->vertexPosition(),trkH->globalTime(),realParent));
	m_vertexMap[iParentID].push_back(MapVertexPosition(m_nVertices,trkH->vertexPosition()));
	m_nVertices++;
	return (m_nVertices-1);
    }
}
 
unsigned int EventAction::g4ToSim(unsigned int i) const
{
    if (i==0) return InvalidID;
    // since I sorted on the first field, I can stop before
    G4ToSimMapType::const_iterator it = m_g4ToSimMap.find(i);
    if (it != m_g4ToSimMap.end()) return (*it).second;
    return InvalidID;
}

unsigned int EventAction::simToG4(unsigned int i) const
{
    if (i<m_simToG4Vector.size()) return m_simToG4Vector[i];
    return InvalidID;
}

void EventAction::cleanVertexMap() { m_vertexMap.clear(); m_nVertices=0; }
