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
// $Id: SimTrackManager.cc,v 1.4 2006/09/11 10:04:03 fambrogl Exp $
//

// system include files
#include <iostream>

// user include files
#include "SimG4Core/Application/interface/SimTrackManager.h"
#include "SimG4Core/Application/interface/G4SimTrack.h"
#include "SimG4Core/Application/interface/G4SimVertex.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


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
   if ( m_trksForThisEvent != 0 ) deleteTracks() ;
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
        edm::LogError("SimG4CoreApplication") << " SimTrackManager::saveTrackAndItsBranch got 0 pointer ";
	//cout << " SimTrackManager::saveTrackAndItsBranch got 0 pointer " << endl;
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
    using namespace std;
    LogDebug("SimTrackManager")  << " SimTrackManager::storeTracks knows " << m_trksForThisEvent->size()
	      << " tracks with history before branching";
    for (unsigned int it =0;  it <(*m_trksForThisEvent).size(); it++)
      LogDebug("SimTrackManager")   << " 1 - Track in position " << it << " G4 track number "
		 << (*m_trksForThisEvent)[it]->trackID()
		 << " mother " << (*m_trksForThisEvent)[it]->parentID()
		 << " status " << (*m_trksForThisEvent)[it]->saved();

    for (unsigned int i = 0; i < m_trksForThisEvent->size(); i++)
      {
	TrackWithHistory * t = (*m_trksForThisEvent)[i];
	if (t->saved()) saveTrackAndItsBranch(i);
      }
    
    // now eliminate from the vector the tracks with only history but not save
    unsigned int num = 0;
    for (unsigned int it = 0;  it < (*m_trksForThisEvent).size(); it++)
      {
	if ((*m_trksForThisEvent)[it]->saved() == true)
	  {
	    if (it>num) (*m_trksForThisEvent)[num] = (*m_trksForThisEvent)[it];
	    num++;
	  }
	else 
	  {	
	    delete (*m_trksForThisEvent)[it];
	  }
      }
    
    (*m_trksForThisEvent).resize(num);
    
    LogDebug("SimTrackManager")  << " AFTER CLEANING, I GET " << (*m_trksForThisEvent).size()
	      << " tracks to be saved persistently";
    
    LogDebug("SimTrackManager")  << "SimTrackManager::storeTracks -  Tracks still alive " 
	      << (*m_trksForThisEvent).size();
    
    for (unsigned int it = 0;  it < (*m_trksForThisEvent).size(); it++)
      LogDebug("SimTrackManager")  << " 3 - Track in position " << it
		<< " G4 track number " << (*m_trksForThisEvent)[it]->trackID()
		<< " mother " << (*m_trksForThisEvent)[it]->parentID()
		<< " status " << (*m_trksForThisEvent)[it]->saved();
    
    stable_sort(m_trksForThisEvent->begin(),m_trksForThisEvent->end(),trkIDLess());
    
    LogDebug("SimTrackManager")   << "SimTrackManager::storeTracks -  TRACKS to be saved starting with "
	       << (*m_trksForThisEvent).size();
    for (unsigned int it = 0;  it < (*m_trksForThisEvent).size(); it++)
      LogDebug("SimTrackManager")   << " 2 - Track in position " << it
		 << " G4 track number " << (*m_trksForThisEvent)[it]->trackID()
		 << " mother " << (*m_trksForThisEvent)[it]->parentID()
		 << " Status " << (*m_trksForThisEvent)[it]->saved();
    
    reallyStoreTracks(simEvent);
}

void SimTrackManager::reallyStoreTracks(G4SimEvent * simEvent)
{
    // loop over the (now ordered) vector and really save the tracks
  LogDebug("SimTrackManager")  << "Inside the reallyStoreTracks method object to be stored = " 
	    << m_trksForThisEvent->size();

    for (unsigned int it = 0; it < m_trksForThisEvent->size(); it++)
    {
        TrackWithHistory * trkH = (*m_trksForThisEvent)[it];
        // at this stage there is one vertex per track, so the vertex id of track N is also N
        int ivertex = -1;
        int ig;

        Hep3Vector pm = 0.;
        unsigned int iParentID = trkH->parentID();
	for(unsigned int iit = 0; iit < m_trksForThisEvent->size(); iit++)
	  {
	    if((*m_trksForThisEvent)[iit]->trackID()==iParentID){
	      pm = (*m_trksForThisEvent)[iit]->momentum();
	      break;
	    }
	  }
        ig = trkH->genParticleID();
        ivertex = getOrCreateVertex(trkH,iParentID,simEvent);
	simEvent->add(new G4SimTrack(trkH->trackID(),trkH->particleID(),
				     trkH->momentum(),trkH->totalEnergy(),ivertex,ig,pm));
    }
}

int SimTrackManager::getOrCreateVertex(TrackWithHistory * trkH, int iParentID,
				       G4SimEvent * simEvent){

  VertexMap::const_iterator iterator = m_vertexMap.find(iParentID);
  if (iterator != m_vertexMap.end()){
    // loop over saved vertices
    for (unsigned int k=0; k<m_vertexMap[iParentID].size(); k++){
      if ((trkH->vertexPosition()-(((m_vertexMap[iParentID])[k]).second)).mag()<0.001)
	return (((m_vertexMap[iParentID])[k]).first);
    }
  }

  int realParent = iParentID;
  
  simEvent->add(new G4SimVertex(trkH->vertexPosition(),trkH->globalTime(),realParent));
  m_vertexMap[iParentID].push_back(MapVertexPosition(m_nVertices,trkH->vertexPosition()));
  m_nVertices++;
  return (m_nVertices-1);

}
 

void SimTrackManager::cleanVertexMap() { m_vertexMap.clear(); m_nVertices=0; }
