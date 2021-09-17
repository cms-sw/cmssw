#ifndef Notification_SimTrackManager_h
#define Notification_SimTrackManager_h
// -*- C++ -*-
//
// Package:     Notification
// Class  :     SimTrackManager
//
/**\class SimTrackManager SimTrackManager.h SimG4Core/Notification/interface/SimTrackManager.h

 Description: Holds tracking information used by the sensitive detectors

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Fri Nov 25 17:36:41 EST 2005
//

// system include files
#include <map>
#include <vector>

// user include files
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/TrackContainer.h"
#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"
#include "FWCore/Utilities/interface/Exception.h"

// forward declarations

class G4SimEvent;

class SimTrackManager {
public:
  class StrictWeakOrdering {
  public:
    bool operator()(TrackWithHistory*& p, const unsigned int& i) const { return p->trackID() < i; }
  };
  //      enum SpecialNumbers {InvalidID = 65535};
  /// this map contains association between vertex number and position
  typedef std::pair<int, math::XYZVectorD> MapVertexPosition;
  typedef std::vector<std::pair<int, math::XYZVectorD> > MapVertexPositionVector;
  typedef std::map<int, MapVertexPositionVector> MotherParticleToVertexMap;
  typedef MotherParticleToVertexMap VertexMap;

  SimTrackManager(bool iCollapsePrimaryVertices = false);
  virtual ~SimTrackManager();

  // ---------- const member functions ---------------------
  const TrackContainer* trackContainer() const { return m_trksForThisEvent; }

  // ---------- member functions ---------------------------
  void storeTracks(G4SimEvent* simEvent);

  void reset();
  void deleteTracks();
  void cleanTkCaloStateInfoMap();

  void cleanTracksWithHistory();

  void addTrack(TrackWithHistory* iTrack, bool inHistory, bool withAncestor) {
    std::pair<int, int> thePair(iTrack->trackID(), iTrack->parentID());
    idsave.push_back(thePair);
    if (inHistory) {
      m_trksForThisEvent->push_back(iTrack);
    }
    if (withAncestor) {
      std::pair<int, int> thisPair(iTrack->trackID(), 0);
      ancestorList.push_back(thisPair);
    }
  }

  void addTkCaloStateInfo(uint32_t t, const std::pair<math::XYZVectorD, math::XYZTLorentzVectorD>& p) {
    std::map<uint32_t, std::pair<math::XYZVectorD, math::XYZTLorentzVectorD> >::const_iterator it =
        mapTkCaloStateInfo.find(t);

    if (it == mapTkCaloStateInfo.end()) {
      mapTkCaloStateInfo.insert(std::pair<uint32_t, std::pair<math::XYZVectorD, math::XYZTLorentzVectorD> >(t, p));
    }
  }
  void setCollapsePrimaryVertices(bool iSet) { m_collapsePrimaryVertices = iSet; }
  int giveMotherNeeded(int i) const {
    int theResult = 0;
    for (unsigned int itr = 0; itr < idsave.size(); itr++) {
      if ((idsave[itr]).first == i) {
        theResult = (idsave[itr]).second;
        break;
      }
    }
    return theResult;
  }
  bool trackExists(unsigned int i) const {
    bool flag = false;
    for (unsigned int itr = 0; itr < (*m_trksForThisEvent).size(); ++itr) {
      if ((*m_trksForThisEvent)[itr]->trackID() == i) {
        flag = true;
        break;
      }
    }
    return flag;
  }
  TrackWithHistory* getTrackByID(unsigned int trackID, bool strict = false) const {
    bool trackFound = false;
    TrackWithHistory* track;
    if (m_trksForThisEvent == nullptr) {
      throw cms::Exception("Unknown", "SimTrackManager") << "m_trksForThisEvent is a nullptr, cannot get any track!";
    }
    for (unsigned int itr = 0; itr < (*m_trksForThisEvent).size(); ++itr) {
      if ((*m_trksForThisEvent)[itr]->trackID() == trackID) {
        track = (*m_trksForThisEvent)[itr];
        trackFound = true;
        break;
      }
    }
    if (!trackFound) {
      if (strict) {
        throw cms::Exception("Unknown", "SimTrackManager")
            << "Attempted to get track " << trackID << " from SimTrackManager, but it's not in m_trksForThisEvent ("
            << (*m_trksForThisEvent).size() << " tracks in m_trksForThisEvent)"
            << "\n";
      }
      return nullptr;
    }
    return track;
  }
  void setLHCTransportLink(const edm::LHCTransportLinkContainer* thisLHCTlink) { theLHCTlink = thisLHCTlink; }

  // stop default
  SimTrackManager(const SimTrackManager&) = delete;
  const SimTrackManager& operator=(const SimTrackManager&) = delete;

private:
  void saveTrackAndItsBranch(TrackWithHistory*);
  int getOrCreateVertex(TrackWithHistory*, int, G4SimEvent* simEvent);
  void cleanVertexMap();
  void reallyStoreTracks(G4SimEvent* simEvent);
  void fillMotherList();
  int idSavedTrack(int) const;

  // to restore the pre-LHC Transport GenParticle id link to a SimTrack
  void resetGenID();

  // ---------- member data --------------------------------
  TrackContainer* m_trksForThisEvent;
  bool m_SaveSimTracks;
  MotherParticleToVertexMap m_vertexMap;
  int m_nVertices;
  bool m_collapsePrimaryVertices;
  std::map<uint32_t, std::pair<math::XYZVectorD, math::XYZTLorentzVectorD> > mapTkCaloStateInfo;
  std::vector<std::pair<int, int> > idsave;

  std::vector<std::pair<int, int> > ancestorList;

  unsigned int lastTrack;
  unsigned int lastHist;

  const edm::LHCTransportLinkContainer* theLHCTlink;
};

class trkIDLess {
public:
  bool operator()(TrackWithHistory* trk1, TrackWithHistory* trk2) const { return (trk1->trackID() < trk2->trackID()); }
};

#endif
