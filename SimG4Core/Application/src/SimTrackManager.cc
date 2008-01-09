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
// $Id: SimTrackManager.cc,v 1.15 2007/12/04 14:41:13 fambrogl Exp $
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
  idsave.reserve(1000);
  niteration=1;
  avgsizeidsave=0;
  sizeidsave=0;
  avgcalomapsize=0;
  calomapsize=0;
  avgvtxmapsize=0;
  vtxmapsize=0;
}


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
void SimTrackManager::reset()
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
  cleanTkCaloStateInfoMap();
  avgsizeidsave+=idsave.size();
  sizeidsave=idsave.size();
  if(niteration>30){
    if(sizeidsave>(avgsizeidsave/niteration)*1.5){
      std::vector<int>().swap(idsave);
      idsave.reserve(1000);
    }else{
      idsave.clear();
    }
  }else{
    idsave.clear();
  }
  niteration++;
}

void SimTrackManager::deleteTracks()
{
    for (unsigned int i = 0; i < m_trksForThisEvent->size(); i++) delete (*m_trksForThisEvent)[i];
    delete m_trksForThisEvent;
    m_trksForThisEvent = 0;
}

/// this saves a track and all its parents looping over the non ordered vector
void SimTrackManager::saveTrackAndItsBranch(TrackWithHistory * trkWHist)
{
    using namespace std;
    TrackWithHistory * trkH = trkWHist;
    if (trkH == 0)
    {
        edm::LogError("SimG4CoreApplication") << " SimTrackManager::saveTrackAndItsBranch got 0 pointer ";
        abort();
    }
    trkH->save();
    unsigned int parent = trkH->parentID();
    bool parentExists=false;

    TrackContainer::const_iterator tk_itr = std::lower_bound((*m_trksForThisEvent).begin(),(*m_trksForThisEvent).end(),
						      parent,SimTrackManager::StrictWeakOrdering());
    TrackWithHistory * tempTk = new TrackWithHistory(**tk_itr);
    if (tk_itr!=m_trksForThisEvent->end() && (*tk_itr)->trackID()==parent) { 
      parentExists=true;  
    }

    if (parentExists) saveTrackAndItsBranch(tempTk);

    delete tempTk;

}

void SimTrackManager::storeTracks(G4SimEvent* simEvent)
{
    using namespace std;

    stable_sort(m_trksForThisEvent->begin(),m_trksForThisEvent->end(),trkIDLess());
    
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
	if (t->saved()) saveTrackAndItsBranch(t);
      }
    
    // now eliminate from the vector the tracks with only history but not save
    unsigned int num = 0;
    for (unsigned int it = 0;  it < (*m_trksForThisEvent).size(); it++)
      {
	int g4ID = (*m_trksForThisEvent)[it]->trackID();
	if ((*m_trksForThisEvent)[it]->saved() == true)
	  {
	    if (it>num) (*m_trksForThisEvent)[num] = (*m_trksForThisEvent)[it];
	    num++;
	    idsave[g4ID] = g4ID;
	  }
	else 
	  {	
	    delete (*m_trksForThisEvent)[it];
	  }
      }
    
    (*m_trksForThisEvent).resize(num);
    
    LogDebug("SimTrackManager")  << " AFTER CLEANING, I GET " << (*m_trksForThisEvent).size()
	      << " tracks to be saved persistently";
    for (unsigned int it = 0;  it < (*m_trksForThisEvent).size(); it++)
      LogDebug("SimTrackManager")   << " Track in position " << it
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

        math::XYZVectorD pm(0.,0.,0.);
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
	std::map<uint32_t,std::pair<math::XYZVectorD,math::XYZTLorentzVectorD> >::const_iterator it = mapTkCaloStateInfo.find(trkH->trackID());
	std::pair<math::XYZVectorD,math::XYZTLorentzVectorD> tcinfo;
	if (it !=  mapTkCaloStateInfo.end()){
	  tcinfo =  it->second;
	}
	simEvent->add(new G4SimTrack(trkH->trackID(),trkH->particleID(),
				     trkH->momentum(),trkH->totalEnergy(),ivertex,ig,pm,tcinfo.first,tcinfo.second));
    }
}

int SimTrackManager::getOrCreateVertex(TrackWithHistory * trkH, int iParentID,
				       G4SimEvent * simEvent){

  int parent = iParentID;
  int check = -1;

  for( std::vector<TrackWithHistory*>::const_iterator it = (*m_trksForThisEvent).begin(); 
       it!= (*m_trksForThisEvent).end();it++){
    if ((*it)->trackID() == uint32_t(parent)){
      check = 0;
      break;
    }
  }

  if(check==-1) parent = -1;

  VertexMap::const_iterator iterator = m_vertexMap.find(parent);
  if (iterator != m_vertexMap.end()){
    // loop over saved vertices
    for (unsigned int k=0; k<m_vertexMap[parent].size(); k++){
      if (sqrt((trkH->vertexPosition()-(((m_vertexMap[parent])[k]).second)).Mag2())<0.001)
	return (((m_vertexMap[parent])[k]).first);
    }
  }
  
  simEvent->add(new G4SimVertex(trkH->vertexPosition(),trkH->globalTime(),parent));
  m_vertexMap[parent].push_back(MapVertexPosition(m_nVertices,trkH->vertexPosition()));
  m_nVertices++;
  return (m_nVertices-1);

}

void SimTrackManager::cleanVertexMap() { 
  avgvtxmapsize+=m_vertexMap.size();
  vtxmapsize=m_vertexMap.size();
  m_vertexMap.clear();
  if(niteration>30){
    if(vtxmapsize>(avgvtxmapsize/niteration)*1.1){
      MotherParticleToVertexMap().swap(m_vertexMap);
    }
  }
  m_nVertices=0; 
}

void SimTrackManager::cleanTkCaloStateInfoMap() { 
  avgcalomapsize+=mapTkCaloStateInfo.size();
  calomapsize=mapTkCaloStateInfo.size();
  mapTkCaloStateInfo.clear();
  if(niteration>30){
    if(calomapsize>(avgcalomapsize/niteration)*1.1){
      std::map<uint32_t,std::pair<math::XYZVectorD,math::XYZTLorentzVectorD > >().swap(mapTkCaloStateInfo);
    }
  }
}

int SimTrackManager::idSavedTrack (int i) const
{
  int id = 0;  
  if (i > 0) {
    id = idsave[i];
    if(id<0)
      id=0;
  }
  return id;
}
