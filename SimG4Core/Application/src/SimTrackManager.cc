// -*- C++ -*-
//
// Package:     Application
// Class  :     SimTrackManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Fri Nov 25 17:44:19 EST 2005
// $Id$
//

// system include files
#include <iostream>

// user include files
#include "SimG4Core/Application/interface/SimTrackManager.h"
#include "SimG4Core/Application/interface/G4SimTrack.h"
#include "SimG4Core/Application/interface/G4SimVertex.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SimTrackManager::SimTrackManager(bool iCollapsePrimaryVertices) :
  m_trksForThisEvent(0),m_nVertices(0),
  m_collapsePrimaryVertices(iCollapsePrimaryVertices)
{
}

// SimTrackManager::SimTrackManager(const SimTrackManager& rhs)
// {
//    // do actual copying here;
// }

SimTrackManager::~SimTrackManager()
{
}

//
// assignment operators
//
// const SimTrackManager& SimTrackManager::operator=(const SimTrackManager& rhs)
// {
//   //An exception safe implementation is
//   SimTrackManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
SimTrackManager::reset()
{
    if (m_trksForThisEvent==0) m_trksForThisEvent = new TrackContainer();
    else
    {
	for (unsigned int i = 0; i < m_trksForThisEvent->size(); i++) 
	    delete (*m_trksForThisEvent)[i];
	delete m_trksForThisEvent;
	m_trksForThisEvent = new TrackContainer();
    }
    cleanVertexMap();
}

void
SimTrackManager::deleteTracks()
{
    for (unsigned int i = 0; i < m_trksForThisEvent->size(); i++) delete (*m_trksForThisEvent)[i];
    delete m_trksForThisEvent;
    m_trksForThisEvent = 0;
}

/// this saves a track and all its parents looping over the non ordered vector
void SimTrackManager::saveTrackAndItsBranch(int i)
{
    using namespace std;
    TrackWithHistory * trkH = (*m_trksForThisEvent)[i];
    if (trkH == 0)
    {
        cout << " SimTrackManager::saveTrackAndItsBranch got 0 pointer " << endl;
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

void SimTrackManager::storeTracks(G4SimEvent* simEvent)
{
#ifdef DEBUG
    using namespace std;
    cout << " SimTrackManager::storeTracks knows " << m_trksForThisEvent->size()
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
    cout << "SimTrackManager::storeTracks -  TRACKS to be saved starting with "
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
    cout << "SimTrackManager::storeTracks -  Tracks still alive " << (*m_trksForThisEvent).size() << endl;
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
    reallyStoreTracks(simEvent);
}

void SimTrackManager::reallyStoreTracks(G4SimEvent * simEvent)
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

int SimTrackManager::getOrCreateVertex(TrackWithHistory * trkH, int iParentID,
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
 
unsigned int SimTrackManager::g4ToSim(unsigned int i) const
{
    if (i==0) return InvalidID;
    // since I sorted on the first field, I can stop before
    G4ToSimMapType::const_iterator it = m_g4ToSimMap.find(i);
    if (it != m_g4ToSimMap.end()) return (*it).second;
    return InvalidID;
}

unsigned int SimTrackManager::simToG4(unsigned int i) const
{
    if (i<m_simToG4Vector.size()) return m_simToG4Vector[i];
    return InvalidID;
}

void SimTrackManager::cleanVertexMap() { m_vertexMap.clear(); m_nVertices=0; }
