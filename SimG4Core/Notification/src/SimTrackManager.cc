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

#include "SimG4Core/Notification/interface/SimTrackManager.h"
#include "SimG4Core/Notification/interface/TmpSimVertex.h"
#include "SimG4Core/Notification/interface/TmpSimEvent.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4VProcess.hh"
#include "G4Track.hh"
#include "G4Event.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"

//#define DebugLog

namespace {
  const double invcm = 1.0 / CLHEP::cm;
  const double invgev = 1.0 / CLHEP::GeV;
  const double r_limit2 = 1.e-6;  // 10 micron in CMS units
  const double t_limit = 10 * CLHEP::picosecond;
}  // namespace

SimTrackManager::SimTrackManager(TmpSimEvent* ptr, int ver) : m_Verbose(ver), m_simEvent(ptr) {
  m_trackContainer = m_simEvent->getHistories();
  m_g4vertices = m_simEvent->getVertices();
}

SimTrackManager::~SimTrackManager() { reset(); }

void SimTrackManager::reset() { m_nPrimary = m_nTracks = m_nVertices = m_nPrimVertices = 0; }

void SimTrackManager::initialisePrimaries(const G4Event* evt) {
  // clean previous event
  reset();
  int nv = evt->GetNumberOfPrimaryVertex();

  // loop over Geant4 primary vertex
  for (int i = 0; i < nv; ++i) {
    auto v = evt->GetPrimaryVertex(i);
    if (nullptr == v) {
      continue;
    }
    math::XYZVectorD pos(v->GetX0() * invcm, v->GetY0() * invcm, v->GetZ0() * invcm);
    double time = v->GetT0();
    int vidx = findOrAddVertex(pos, time, 0, 0);

    // primary particle for a vertex
    int np = v->GetNumberOfParticle();
    for (int j = 0; j < np; ++j) {
      auto prim = v->GetPrimary(j);
      if (nullptr == prim) {
        continue;
      }
      ++m_nPrimary;
      auto p = new TrackWithHistory(prim, m_nPrimary, pos, time);
      m_simEvent->addTrack(p);
      ++m_nTracks;
      p->setVertexID(vidx);
    }
  }
  m_nPrimVertices = m_nVertices;
  if (m_Verbose > 1) {
    edm::LogVerbatim("SimG4CoreNotification")
        << "SimTrackManager::initialisePrimaries() Ntracks=" << m_nPrimary << " nVertex=" << m_nVertices;
  }
}

int SimTrackManager::findOrAddVertex(math::XYZVectorD& pos, double& time, int i1, int i2) {
  // check if the vertex coincide with previous
  bool isNew = true;
  int vidx = m_nVertices;
  for (int j = 0; j < m_nVertices; ++j) {
    if (((*m_g4vertices)[j]->vertexPosition() - pos).Mag2() < r_limit2 &&
        std::abs((*m_g4vertices)[j]->vertexGlobalTime() - time) < t_limit) {
      isNew = false;
      m_simVertex = (*m_g4vertices)[j];
      pos = m_simVertex->vertexPosition();
      time = m_simVertex->vertexGlobalTime();
      vidx = j;
      break;
    }
  }
  // create new vertex
  if (isNew) {
    m_simVertex = new TmpSimVertex(pos, time, vidx, i1, i2);
    m_simEvent->addVertex(m_simVertex);
    ++m_nVertices;
  }
  return vidx;
}

TrackWithHistory* SimTrackManager::getTrackWithHistory(const G4Track* track) {
  m_currTrack = track;
  m_currTrackInfo = static_cast<TrackInformation*>(track->GetUserInformation());

  // track are produced by generator no real history yet
  // index in the track vector is (id - 1)
  int id = track->GetTrackID();

  if (m_Verbose > 2) {
    edm::LogVerbatim("SimG4CoreNotification")
        << "SimTrackManager::getTrackWithHistory id=" << id << " nTracks=" << m_simEvent->nTracks()
        << " nPrimary=" << m_nPrimary << " nVertex=" << m_nVertices << " MCtruthID=" << m_currTrackInfo->mcTruthID();
  }
  if (id <= m_nPrimary && id > 0) {
    // use primary history
    m_currHistory = m_simEvent->getHistory(id - 1);

  } else {
    // new history
    int mc = m_currTrackInfo->mcTruthID();
    if (m_currTrackInfo->storeTrack()) {
      mc = m_nTracks;
    }
    m_currHistory = new TrackWithHistory(track, mc);
  }
  return m_currHistory;
}

void SimTrackManager::addTrack(bool inHistory) {
  // check if track should be stored
  if (!inHistory && !m_currHistory->crossedBoundary()) {
    delete m_currHistory;
    return;
  }
  int id = m_currTrack->GetTrackID();
  int parid = m_currHistory->mcTruthID();
  if (m_Verbose > 2) {
    edm::LogVerbatim("SimG4CoreNotification") << "SimTrackManager::addTrack id=" << id << " parentID=" << parid
                                              << " Ekin(MeV)=" << m_currTrack->GetVertexKineticEnergy() / CLHEP::GeV
                                              << " vertexID=" << m_currHistory->vertexID();
  }
  if (!inHistory && m_currHistory->crossedBoundary()) {
    if (parid < m_nTracks) {
      auto parent = (*m_trackContainer)[parid];
      if (parent->crossedBoundary()) {
        delete m_currHistory;
        return;
      }
      parent->setCrossedBoundaryPosMom(
          id, m_currTrackInfo->getPositionAtBoundary(), m_currTrackInfo->getMomentumAtBoundary());
    } else {
      delete m_currHistory;
      return;
    }
  }

  // vertex is not yet defined
  if (m_currHistory->vertexID() < 0) {
    auto& p = m_currTrack->GetVertexPosition();
    math::XYZVectorD pos(p.x() * invcm, p.y() * invcm, p.z() * invcm);
    double time = m_currHistory->time();
    int vidx = findOrAddVertex(pos, time, parid, m_currHistory->processType());
    m_currHistory->setVertexID(vidx);
  }

  // primary are already added
  if (id > m_nPrimary) {
    m_simEvent->addTrack(m_currHistory);
    ++m_nTracks;
  }

  const auto sp = m_currTrack->GetStep()->GetPostStepPoint();
  const auto& p = sp->GetPosition();
  const auto& v = sp->GetMomentum();
  double e = sp->GetTotalEnergy() * invgev;

  m_currHistory->setSurfacePosMom(math::XYZVectorD(p.x() * invcm, p.y() * invcm, p.z() * invcm),
                                  math::XYZTLorentzVectorD(v.x() * invgev, v.y() * invgev, v.z() * invgev, e));
}

void SimTrackManager::storeTracks() {}
