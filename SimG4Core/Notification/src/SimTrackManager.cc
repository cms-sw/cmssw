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
#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4VProcess.hh"
#include "G4Track.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"

//#define DebugLog

SimTrackManager::SimTrackManager() {
  idsave.reserve(1000);
  ancestorList.reserve(1000);
  m_trackContainer.reserve(1000);
  m_endPoints.reserve(1000);
}

SimTrackManager::~SimTrackManager() { reset(); }

void SimTrackManager::reset() {
  deleteTracks();
  cleanVertexMap();
  idsave.clear();
  ancestorList.clear();
  lastTrack = 0;
  lastHist = 0;
}

void SimTrackManager::deleteTracks() {
  if (!m_trackContainer.empty()) {
    for (auto& ptr : m_trackContainer) {
      delete ptr;
    }
    m_trackContainer.clear();
    m_endPoints.clear();
  }
}

void SimTrackManager::cleanVertexMap() {
  m_vertexMap.clear();
  m_vertexMap.swap(m_vertexMap);
  m_nVertices = 0;
}

void SimTrackManager::addTrack(TrackWithHistory* iTrack, const G4Track* track, bool inHistory, bool withAncestor) {
  std::pair<int, int> thePair(iTrack->trackID(), iTrack->parentID());
  idsave.push_back(thePair);
  if (inHistory) {
    m_trackContainer.push_back(iTrack);
    const auto& v = track->GetStep()->GetPostStepPoint()->GetPosition();
    const double invcm = 1.0 / CLHEP::cm;
    std::pair<int, math::XYZVectorD> p(iTrack->trackID(),
                                       math::XYZVectorD(v.x() * invcm, v.y() * invcm, v.z() * invcm));
    m_endPoints.push_back(p);
  }
  if (withAncestor) {
    std::pair<int, int> thisPair(iTrack->trackID(), 0);
    ancestorList.push_back(thisPair);
  }
}

/// this saves a track and all its parents looping over the non ordered vector
void SimTrackManager::saveTrackAndItsBranch(TrackWithHistory* trkWHist) {
  TrackWithHistory* trkH = trkWHist;
  if (trkH == nullptr) {
    edm::LogError("SimTrackManager") << " SimTrackManager::saveTrackAndItsBranch got 0 pointer ";
    throw cms::Exception("SimTrackManager::saveTrackAndItsBranch") << " cannot handle hits for tracking";
  }
  trkH->setToBeSaved();
  unsigned int parent = trkH->parentID();

  auto tk_itr =
      std::lower_bound(m_trackContainer.begin(), m_trackContainer.end(), parent, SimTrackManager::StrictWeakOrdering());

  if (tk_itr != m_trackContainer.end() && (*tk_itr)->trackID() == parent) {
    saveTrackAndItsBranch(*tk_itr);
  }
}

void SimTrackManager::storeTracks(G4SimEvent* simEvent) {
  cleanTracksWithHistory();

  // fill the map with the final mother-daughter relationship
  idsave.swap(ancestorList);
  std::stable_sort(idsave.begin(), idsave.end());

  std::vector<std::pair<int, int> >().swap(ancestorList);

  // to get a backward compatible order
  std::stable_sort(m_trackContainer.begin(), m_trackContainer.end(), trkIDLess());

  // to reset the GenParticle ID of a SimTrack to its pre-LHCTransport value
  resetGenID();

  reallyStoreTracks(simEvent);
}

void SimTrackManager::reallyStoreTracks(G4SimEvent* simEvent) {
  // loop over the (now ordered) vector and really save the tracks
#ifdef DebugLog
  edm::LogVerbatim("SimTrackManager") << "reallyStoreTracks() NtracksWithHistory= " << m_trackContainer->size();
#endif

  int nn = m_endPoints.size();
  for (auto& trkH : m_trackContainer) {
    // at this stage there is one vertex per track,
    // so the vertex id of track N is also N
    unsigned int iParentID = trkH->parentID();
    int ig = trkH->genParticleID();
    int ivertex = getOrCreateVertex(trkH, iParentID, simEvent);

    auto ptr = trkH;
    if (0 < iParentID) {
      for (auto& trk : m_trackContainer) {
        if (trk->trackID() == iParentID) {
          ptr = trk;
          break;
        }
      }
    }
    // Track at surface is the track at intersection point between tracker and calo
    // envelops if exist. If not exist the momentum is zero, position is the end of
    // the track
    const math::XYZVectorD& pm = ptr->momentum();
    math::XYZVectorD spos(0., 0., 0.);
    math::XYZTLorentzVectorD smom(0., 0., 0., 0.);
    int id = trkH->trackID();
    if (trkH->crossedBoundary()) {
      spos = trkH->getPositionAtBoundary();
      smom = trkH->getMomentumAtBoundary();
    } else {
      for (int i = 0; i < nn; ++i) {
        if (id == m_endPoints[i].first) {
          spos = m_endPoints[i].second;
          break;
        }
      }
    }

    G4SimTrack* g4simtrack =
        new G4SimTrack(id, trkH->particleID(), trkH->momentum(), trkH->totalEnergy(), ivertex, ig, pm, spos, smom);
    g4simtrack->copyCrossedBoundaryVars(trkH);
    simEvent->add(g4simtrack);
  }
}

int SimTrackManager::getOrCreateVertex(TrackWithHistory* trkH, int iParentID, G4SimEvent* simEvent) {
  int parent = -1;
  for (auto& trk : m_trackContainer) {
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
  m_vertexMap[parent].push_back(VertexPosition(m_nVertices, trkH->vertexPosition()));
  ++m_nVertices;
  return (m_nVertices - 1);
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
#ifdef DebugLog
  edm::LogVerbatim("SimTrackManager") << "### SimTrackManager::fillMotherList: " << idsave.size()
                                      << " saved; ancestor: " << lastHist << "  " << ancestorList.size();
  for (unsigned int i = 0; i < idsave.size(); ++i) {
    edm::LogVerbatim("SimTrackManager") << " ISV: Track ID = " << (idsave[i]).first
                                        << " Mother ID = " << (idsave[i]).second;
  }
#endif
  for (unsigned int n = lastHist; n < ancestorList.size(); ++n) {
    int theMotherId = idSavedTrack((ancestorList[n]).first);
    ancestorList[n].second = theMotherId;
#ifdef DebugLog
    LogDebug("SimTrackManager") << "Track ID = " << (ancestorList[n]).first
                                << " Mother ID = " << (ancestorList[n]).second;
#endif
  }

  lastHist = ancestorList.size();
  idsave.clear();
}

void SimTrackManager::cleanTracksWithHistory() {
  if (m_trackContainer.empty() && idsave.empty()) {
    return;
  }

#ifdef DebugLog
  LogDebug("SimTrackManager") << "SimTrackManager::cleanTracksWithHistory has " << idsave.size()
                              << " mother-daughter relationships stored with lastTrack = " << lastTrack;
#endif

  if (lastTrack > 0 && lastTrack >= m_trackContainer.size()) {
    lastTrack = 0;
    edm::LogError("SimTrackManager") << " SimTrackManager::cleanTracksWithHistory track index corrupted";
  }

  std::stable_sort(m_trackContainer.begin() + lastTrack, m_trackContainer.end(), trkIDLess());
  std::stable_sort(idsave.begin(), idsave.end());

#ifdef DebugLog
  LogDebug("SimTrackManager") << " SimTrackManager::cleanTracksWithHistory knows " << m_trksForThisEvent->size()
                              << " tracks with history before branching";
  for (unsigned int it = 0; it < m_trackContainer.size(); it++) {
    LogDebug("SimTrackManager") << " 1 - Track in position " << it << " G4 track number "
                                << m_trackContainer[it]->trackID() << " mother " << m_trackContainer[it]->parentID()
                                << " status " << m_trackContainer[it]->saved();
  }
#endif

  for (auto& t : m_trackContainer) {
    if (t->saved()) {
      saveTrackAndItsBranch(t);
    }
  }
  unsigned int num = lastTrack;
  for (unsigned int it = lastTrack; it < m_trackContainer.size(); ++it) {
    auto t = m_trackContainer[it];
    int g4ID = m_trackContainer[it]->trackID();
    if (t->saved()) {
      if (it > num) {
        m_trackContainer[num] = t;
      }
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

  m_trackContainer.resize(num);

#ifdef DebugLog
  LogDebug("SimTrackManager") << " AFTER CLEANING, I GET " << m_trackContainer.size()
                              << " tracks to be saved persistently";
  for (unsigned int it = 0; it < m_trackContainer.size(); ++it) {
    LogDebug("SimTrackManager") << " Track in position " << it << " G4 track number " << m_trackContainer[it]->trackID()
                                << " mother " << m_trackContainer[it]->parentID() << " Status "
                                << m_trackContainer[it]->saved() << " id " << m_trackContainer[it]->particleID()
                                << " E(MeV)= " << m_trackContainer[it]->totalEnergy();
  }
#endif

  fillMotherList();
  lastTrack = m_trackContainer.size();
}

void SimTrackManager::resetGenID() {
  if (theLHCTlink == nullptr)
    return;

  for (auto& trkH : m_trackContainer) {
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

void SimTrackManager::ReportException(unsigned int id) const {
  throw cms::Exception("Unknown", "SimTrackManager::getTrackByID")
      << "Fail to get track " << id << " from SimTrackManager, container size= " << m_trackContainer.size();
}
