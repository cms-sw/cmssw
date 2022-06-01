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
//

// system include files
#include <iostream>

// user include files
#include "SimG4Core/Notification/interface/SimTrackManager.h"
#include "SimG4Core/Notification/interface/G4SimTrack.h"
#include "SimG4Core/Notification/interface/G4SimVertex.h"
#include "SimG4Core/Notification/interface/G4SimEvent.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4VProcess.hh"

//#define DebugLog

SimTrackManager::SimTrackManager(bool iCollapsePrimaryVertices)
    : m_trksForThisEvent(nullptr),
      m_nVertices(0),
      m_collapsePrimaryVertices(iCollapsePrimaryVertices),
      lastTrack(0),
      lastHist(0),
      theLHCTlink(nullptr) {}

SimTrackManager::~SimTrackManager() {
  if (m_trksForThisEvent != nullptr)
    deleteTracks();
}

//
// member functions
//
void SimTrackManager::reset() {
  if (m_trksForThisEvent == nullptr) {
    m_trksForThisEvent = new TrackContainer();
  } else {
    for (unsigned int i = 0; i < m_trksForThisEvent->size(); i++) {
      delete (*m_trksForThisEvent)[i];
    }
    delete m_trksForThisEvent;
    m_trksForThisEvent = new TrackContainer();
  }
  cleanVertexMap();
  cleanTkCaloStateInfoMap();
  std::vector<std::pair<int, int> >().swap(idsave);
  ancestorList.clear();
  lastTrack = 0;
  lastHist = 0;
}

void SimTrackManager::deleteTracks() {
  for (unsigned int i = 0; i < m_trksForThisEvent->size(); ++i) {
    delete (*m_trksForThisEvent)[i];
  }
  delete m_trksForThisEvent;
  m_trksForThisEvent = nullptr;
}

/// this saves a track and all its parents looping over the non ordered vector
void SimTrackManager::saveTrackAndItsBranch(TrackWithHistory* trkWHist) {
  using namespace std;
  TrackWithHistory* trkH = trkWHist;
  if (trkH == nullptr) {
    edm::LogError("SimTrackManager") << " SimTrackManager::saveTrackAndItsBranch got 0 pointer ";
    throw cms::Exception("SimTrackManager::saveTrackAndItsBranch") << " cannot handle hits for tracking";
  }
  trkH->save();
  unsigned int parent = trkH->parentID();

  TrackContainer::const_iterator tk_itr = std::lower_bound(
      (*m_trksForThisEvent).begin(), (*m_trksForThisEvent).end(), parent, SimTrackManager::StrictWeakOrdering());

  if (tk_itr != m_trksForThisEvent->end() && (*tk_itr)->trackID() == parent) {
    saveTrackAndItsBranch(*tk_itr);
  }
}

void SimTrackManager::storeTracks(G4SimEvent* simEvent) {
  cleanTracksWithHistory();

  // fill the map with the final mother-daughter relationship
  idsave.swap(ancestorList);
  stable_sort(idsave.begin(), idsave.end());

  std::vector<std::pair<int, int> >().swap(ancestorList);

  // to get a backward compatible order
  stable_sort(m_trksForThisEvent->begin(), m_trksForThisEvent->end(), trkIDLess());

  // to reset the GenParticle ID of a SimTrack to its pre-LHCTransport value
  resetGenID();

  reallyStoreTracks(simEvent);
}

void SimTrackManager::reallyStoreTracks(G4SimEvent* simEvent) {
  // loop over the (now ordered) vector and really save the tracks
#ifdef DebugLog
  LogDebug("SimTrackManager") << "Inside the reallyStoreTracks method object to be stored = "
                              << m_trksForThisEvent->size();
#endif

  for (auto& trkH : *m_trksForThisEvent) {
    // at this stage there is one vertex per track,
    // so the vertex id of track N is also N
    int ivertex = -1;
    int ig;

    math::XYZVectorD pm(0., 0., 0.);
    unsigned int iParentID = trkH->parentID();
    for (auto& trk : *m_trksForThisEvent) {
      if (trk->trackID() == iParentID) {
        pm = trk->momentum();
        break;
      }
    }
    ig = trkH->genParticleID();
    ivertex = getOrCreateVertex(trkH, iParentID, simEvent);
    std::map<uint32_t, std::pair<math::XYZVectorD, math::XYZTLorentzVectorD> >::const_iterator cit =
        mapTkCaloStateInfo.find(trkH->trackID());
    std::pair<math::XYZVectorD, math::XYZTLorentzVectorD> tcinfo;
    if (cit != mapTkCaloStateInfo.end()) {
      tcinfo = cit->second;
    }
    G4SimTrack* g4simtrack = new G4SimTrack(trkH->trackID(),
                                            trkH->particleID(),
                                            trkH->momentum(),
                                            trkH->totalEnergy(),
                                            ivertex,
                                            ig,
                                            pm,
                                            tcinfo.first,
                                            tcinfo.second);
    g4simtrack->copyCrossedBoundaryVars(trkH);
    simEvent->add(g4simtrack);
  }
}

int SimTrackManager::getOrCreateVertex(TrackWithHistory* trkH, int iParentID, G4SimEvent* simEvent) {
  int parent = -1;
  for (auto& trk : *m_trksForThisEvent) {
    int id = trk->trackID();
    if (id == iParentID) {
      parent = id;
      break;
    }
  }

  VertexMap::const_iterator iterator = m_vertexMap.find(parent);
  if (iterator != m_vertexMap.end()) {
    // loop over saved vertices
    for (auto& xx : m_vertexMap[parent]) {
      if ((trkH->vertexPosition() - xx.second).Mag2() < 1.e-6) {
        return xx.first;
      }
    }
  }

  unsigned int ptype = 0;
  const G4VProcess* pr = trkH->creatorProcess();
  if (nullptr != pr) {
    ptype = pr->GetProcessSubType();
  }
  simEvent->add(new G4SimVertex(trkH->vertexPosition(), trkH->globalTime(), parent, ptype));
  m_vertexMap[parent].push_back(MapVertexPosition(m_nVertices, trkH->vertexPosition()));
  ++m_nVertices;
  return (m_nVertices - 1);
}

void SimTrackManager::cleanVertexMap() {
  m_vertexMap.clear();
  MotherParticleToVertexMap().swap(m_vertexMap);
  m_nVertices = 0;
}

void SimTrackManager::cleanTkCaloStateInfoMap() {
  mapTkCaloStateInfo.clear();
  std::map<uint32_t, std::pair<math::XYZVectorD, math::XYZTLorentzVectorD> >().swap(mapTkCaloStateInfo);
}

int SimTrackManager::idSavedTrack(int id) const {
  int idMother = id;
  if (id > 0) {
    unsigned int n = idsave.size();
    if (0 < n) {
      int jmax = n - 1;
      int j, id1;

      // first loop forward
      bool notFound = true;
      for (j = 0; j <= jmax; ++j) {
        if ((idsave[j]).first == idMother) {
          id1 = (idsave[j]).second;
          if (0 == id1 || id1 == idMother) {
            return id1;
          }
          jmax = j - 1;
          idMother = id1;
          notFound = false;
          break;
        }
      }
      if (notFound) {
        return 0;
      }

      // recursive loop
      do {
        notFound = true;
        // search ID scan backward
        for (j = jmax; j >= 0; --j) {
          if ((idsave[j]).first == idMother) {
            id1 = (idsave[j]).second;
            if (0 == id1 || id1 == idMother) {
              return id1;
            }
            jmax = j - 1;
            idMother = id1;
            notFound = false;
            break;
          }
        }
        if (notFound) {
          // ID not in the list of saved track - look into ancestors
          jmax = ancestorList.size() - 1;
          for (j = jmax; j >= 0; --j) {
            if ((ancestorList[j]).first == idMother) {
              idMother = (ancestorList[j]).second;
              return idMother;
            }
          }
          return 0;
        }
      } while (!notFound);
    }
  }
  return idMother;
}

void SimTrackManager::fillMotherList() {
  if (!ancestorList.empty() && lastHist > ancestorList.size()) {
    lastHist = ancestorList.size();
    edm::LogError("SimTrackManager") << " SimTrackManager::fillMotherList track index corrupted";
  }
  /*
  edm::LogVerbatim("SimTrackManager") << "### SimTrackManager::fillMotherList: "
	    << idsave.size() << " saved; ancestor: " << lastHist 
	    << "  " << ancestorList.size() << std::endl;
  for (unsigned int i = 0; i< idsave.size(); ++i) { 
    edm::LogVerbatim("SimTrackManager")  << " ISV: Track ID = " << (idsave[i]).first 
	       << " Mother ID = " << (idsave[i]).second << std::endl;
  }
  */
  for (unsigned int n = lastHist; n < ancestorList.size(); ++n) {
    int theMotherId = idSavedTrack((ancestorList[n]).first);
    ancestorList[n].second = theMotherId;
    /*
    std::cout  << " ANC: Track ID = " << (ancestorList[n]).first 
	       << " Mother ID = " << (ancestorList[n]).second << std::endl;
    */
#ifdef DebugLog
    LogDebug("SimTrackManager") << "Track ID = " << (ancestorList[n]).first
                                << " Mother ID = " << (ancestorList[n]).second;
#endif
  }

  lastHist = ancestorList.size();

  idsave.clear();
}

void SimTrackManager::cleanTracksWithHistory() {
  if ((*m_trksForThisEvent).empty() && idsave.empty()) {
    return;
  }

#ifdef DebugLog
  LogDebug("SimTrackManager") << "SimTrackManager::cleanTracksWithHistory has " << idsave.size()
                              << " mother-daughter relationships stored with lastTrack = " << lastTrack;
#endif

  if (lastTrack > 0 && lastTrack >= (*m_trksForThisEvent).size()) {
    lastTrack = 0;
    edm::LogError("SimTrackManager") << " SimTrackManager::cleanTracksWithHistory track index corrupted";
  }

  stable_sort(m_trksForThisEvent->begin() + lastTrack, m_trksForThisEvent->end(), trkIDLess());

  stable_sort(idsave.begin(), idsave.end());

#ifdef DebugLog
  LogDebug("SimTrackManager") << " SimTrackManager::cleanTracksWithHistory knows " << m_trksForThisEvent->size()
                              << " tracks with history before branching";
  for (unsigned int it = 0; it < (*m_trksForThisEvent).size(); it++) {
    LogDebug("SimTrackManager") << " 1 - Track in position " << it << " G4 track number "
                                << (*m_trksForThisEvent)[it]->trackID() << " mother "
                                << (*m_trksForThisEvent)[it]->parentID() << " status "
                                << (*m_trksForThisEvent)[it]->saved();
  }
#endif

  for (auto& t : *m_trksForThisEvent) {
    if (t->saved()) {
      saveTrackAndItsBranch(t);
    }
  }
  unsigned int num = lastTrack;
  for (unsigned int it = lastTrack; it < m_trksForThisEvent->size(); ++it) {
    TrackWithHistory* t = (*m_trksForThisEvent)[it];
    int g4ID = t->trackID();
    if (t->saved() == true) {
      if (it > num)
        (*m_trksForThisEvent)[num] = t;
      ++num;
      for (auto& xx : idsave) {
        if (xx.first == g4ID) {
          xx.second = g4ID;
          break;
        }
      }
    } else {
      delete t;
    }
  }

  m_trksForThisEvent->resize(num);

#ifdef DebugLog
  LogDebug("SimTrackManager") << " AFTER CLEANING, I GET " << (*m_trksForThisEvent).size()
                              << " tracks to be saved persistently";
  for (unsigned int it < (*m_trksForThisEvent).size(); ++it) {
    LogDebug("SimTrackManager") << " Track in position " << it << " G4 track number "
                                << (*m_trksForThisEvent)[it]->trackID() << " mother "
                                << (*m_trksForThisEvent)[it]->parentID() << " Status "
                                << (*m_trksForThisEvent)[it]->saved() << " id "
                                << (*m_trksForThisEvent)[it]->particleID()
                                << " E(MeV)= " << (*m_trksForThisEvent)[it]->totalEnergy();
  }
#endif

  fillMotherList();

  lastTrack = (*m_trksForThisEvent).size();
}

void SimTrackManager::resetGenID() {
  if (theLHCTlink == nullptr)
    return;

  for (auto& trkH : *m_trksForThisEvent) {
    int genParticleID = trkH->genParticleID();
    if (genParticleID == -1) {
      continue;
    } else {
      for (auto& xx : *theLHCTlink) {
        if (xx.afterHector() == genParticleID) {
          trkH->setGenParticleID(xx.beforeHector());
          continue;
        }
      }
    }
  }

  theLHCTlink = nullptr;
}
