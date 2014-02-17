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
// $Id: SimTrackManager.cc,v 1.21 2012/04/02 12:24:43 davidlt Exp $
//

// system include files
#include <iostream>

// user include files
#include "SimG4Core/Application/interface/SimTrackManager.h"
#include "SimG4Core/Application/interface/G4SimTrack.h"
#include "SimG4Core/Application/interface/G4SimVertex.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define DebugLog

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
  m_collapsePrimaryVertices(iCollapsePrimaryVertices),
  lastTrack(0),lastHist(0),theLHCTlink(0){}


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
  std::vector<std::pair <int, int> >().swap(idsave);
  ancestorList.clear();
  lastTrack=0;
  lastHist=0;
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
      edm::LogError("SimTrackManager") << " SimTrackManager::saveTrackAndItsBranch got 0 pointer ";
      abort();
    }
  trkH->save();
  unsigned int parent = trkH->parentID();
  bool parentExists=false;
  
  TrackContainer::const_iterator tk_itr = std::lower_bound((*m_trksForThisEvent).begin(),(*m_trksForThisEvent).end(),
                                                           parent,SimTrackManager::StrictWeakOrdering());

  TrackWithHistory * tempTk = *tk_itr;
  //  TrackWithHistory * tempTk = new TrackWithHistory(**tk_itr);
  if (tk_itr!=m_trksForThisEvent->end() && (*tk_itr)->trackID()==parent) { 
    parentExists=true;  
  }
  
  if (parentExists) saveTrackAndItsBranch(tempTk);
  
  //  delete tempTk;
  
}

void SimTrackManager::storeTracks(G4SimEvent* simEvent)
{
  cleanTracksWithHistory();

  // fill the map with the final mother-daughter relationship
  idsave.swap(ancestorList);
  stable_sort(idsave.begin(),idsave.end());

  std::vector<std::pair<int,int> >().swap(ancestorList);

  // to get a backward compatible order
  stable_sort(m_trksForThisEvent->begin(),m_trksForThisEvent->end(),trkIDLess());

  // to reset the GenParticle ID of a SimTrack to its pre-LHCTransport value
  resetGenID();

  reallyStoreTracks(simEvent);
}

void SimTrackManager::reallyStoreTracks(G4SimEvent * simEvent)
{
  // loop over the (now ordered) vector and really save the tracks
#ifdef DebugLog
  LogDebug("SimTrackManager")  << "Inside the reallyStoreTracks method object to be stored = " 
                               << m_trksForThisEvent->size();
#endif 
 
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
      std::map<uint32_t,std::pair<math::XYZVectorD,math::XYZTLorentzVectorD> >::const_iterator cit = mapTkCaloStateInfo.find(trkH->trackID());
      std::pair<math::XYZVectorD,math::XYZTLorentzVectorD> tcinfo;
      if (cit !=  mapTkCaloStateInfo.end()){
        tcinfo = cit->second;
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
  m_vertexMap.clear();
  MotherParticleToVertexMap().swap(m_vertexMap);
  m_nVertices=0; 
}

void SimTrackManager::cleanTkCaloStateInfoMap() { 
  mapTkCaloStateInfo.clear();
  std::map<uint32_t,std::pair<math::XYZVectorD,math::XYZTLorentzVectorD > >().swap(mapTkCaloStateInfo);
}

int SimTrackManager::idSavedTrack (int i) const
{

  int id = 0;  
  if (i > 0) {
    for (unsigned int itr=0; itr<idsave.size(); itr++) { if ((idsave[itr]).first == i) { id = (idsave[itr]).second; break; } }
      if (id != i) return idSavedTrack(id);
      id = i;
  }
  return id;
}


void SimTrackManager::fillMotherList() {

  if ( ancestorList.size() > 0 && lastHist > ancestorList.size() ) {
    lastHist = ancestorList.size();
    edm::LogError("SimTrackManager") << " SimTrackManager::fillMotherList track index corrupted";
  }

  for (unsigned int n = lastHist; n < ancestorList.size(); n++) { 
    
    int theMotherId = idSavedTrack((ancestorList[n]).first);
    ancestorList[n].second = theMotherId;
#ifdef DebugLog
    LogDebug("SimTrackManager")  << "Track ID = " << (ancestorList[n]).first << " Mother ID = " << (ancestorList[n]).second;
#endif    
  }

  lastHist = ancestorList.size();

  idsave.clear();

}

void SimTrackManager::cleanTracksWithHistory(){

  using namespace std;

  if ((*m_trksForThisEvent).size() == 0 && idsave.size() == 0) return;

#ifdef DebugLog
  LogDebug("SimTrackManager") << "SimTrackManager::cleanTracksWithHistory has " << idsave.size() 
                              << " mother-daughter relationships stored with lastTrack = " << lastTrack;
#endif

  if ( lastTrack > 0 && lastTrack >= (*m_trksForThisEvent).size() ) {
    lastTrack = 0;
    edm::LogError("SimTrackManager") << " SimTrackManager::cleanTracksWithHistory track index corrupted";
  }
  
  stable_sort(m_trksForThisEvent->begin()+lastTrack,m_trksForThisEvent->end(),trkIDLess());
  
  stable_sort(idsave.begin(),idsave.end());
  
#ifdef DebugLog
  LogDebug("SimTrackManager")  << " SimTrackManager::cleanTracksWithHistory knows " << m_trksForThisEvent->size()
                               << " tracks with history before branching";
  for (unsigned int it =0;  it <(*m_trksForThisEvent).size(); it++)
    LogDebug("SimTrackManager")   << " 1 - Track in position " << it << " G4 track number "
                                  << (*m_trksForThisEvent)[it]->trackID()
                                  << " mother " << (*m_trksForThisEvent)[it]->parentID()
                                  << " status " << (*m_trksForThisEvent)[it]->saved();
#endif  

  for (unsigned int it = lastTrack; it < m_trksForThisEvent->size(); it++)
    {
      TrackWithHistory * t = (*m_trksForThisEvent)[it];
      if (t->saved()) saveTrackAndItsBranch(t);
    }
  unsigned int num = lastTrack;
  for (unsigned int it = lastTrack; it < m_trksForThisEvent->size(); it++)
    {
      TrackWithHistory * t = (*m_trksForThisEvent)[it];
      int g4ID = t->trackID();
      if (t->saved() == true)
        {
          if (it>num) (*m_trksForThisEvent)[num] = t;
          num++;
	  for (unsigned int itr=0; itr<idsave.size(); itr++) { 
	    if ((idsave[itr]).first == g4ID) { 
	      (idsave[itr]).second = g4ID; break; } 
	  }
        }
      else 
        {	
          delete t;
        }
    }
  
  (*m_trksForThisEvent).resize(num);
  
#ifdef DebugLog
  LogDebug("SimTrackManager")  << " AFTER CLEANING, I GET " << (*m_trksForThisEvent).size()
                               << " tracks to be saved persistently";
  for (unsigned int it = 0;  it < (*m_trksForThisEvent).size(); it++)
    LogDebug("SimTrackManager")   << " Track in position " << it
                                  << " G4 track number " << (*m_trksForThisEvent)[it]->trackID()
                                  << " mother " << (*m_trksForThisEvent)[it]->parentID()
                                  << " Status " << (*m_trksForThisEvent)[it]->saved();
#endif  

  fillMotherList();

  lastTrack = (*m_trksForThisEvent).size();

}

void SimTrackManager::resetGenID() {

  if ( theLHCTlink == 0 ) return;

  for  (unsigned int it = 0; it < m_trksForThisEvent->size(); it++)
    {
      TrackWithHistory * trkH = (*m_trksForThisEvent)[it];
      int genParticleID_ = trkH->genParticleID();
      if ( genParticleID_ == -1 ) { continue; }
      else {
        for ( unsigned int itrlink = 0; itrlink < (*theLHCTlink).size(); itrlink++ ) {
          if ( (*theLHCTlink)[itrlink].afterHector() == genParticleID_ ) {
            trkH->setGenParticleID( (*theLHCTlink)[itrlink].beforeHector() );
            continue;
          }
        }
      }
    }

  theLHCTlink = 0;

}
