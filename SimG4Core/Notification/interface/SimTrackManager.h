#ifndef Notification_SimTrackManager_h
#define Notification_SimTrackManager_h
// -*- C++ -*-
//
// Package:     Notification
// Class  :     SimTrackManager
//
/**\class SimTrackManager SimTrackManager.h SimG4Core/Notification/interface/SimTrackManager.h

 Description: Holds tracking information used by the sensitive detectors
 Created:  Fri Nov 25 17:36:41 EST 2005

*/

// system include files
#include <map>
#include <vector>

// user include files
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"

// forward declarations

class G4SimEvent;
class G4Track;

class SimTrackManager {
public:
  class StrictWeakOrdering {
  public:
    bool operator()(TrackWithHistory*& p, const unsigned int& i) const { return p->trackID() < i; }
  };

  typedef std::pair<int, math::XYZVectorD> VertexPosition;
  typedef std::vector<std::pair<int, math::XYZVectorD> > VertexPositionVector;
  typedef std::map<int, VertexPositionVector> VertexMap;

  SimTrackManager();
  virtual ~SimTrackManager();

  const std::vector<TrackWithHistory*>* trackContainer() const { return &m_trackContainer; }

  void storeTracks(G4SimEvent* simEvent);

  void reset();
  void deleteTracks();
  void cleanTracksWithHistory();

  void addTrack(TrackWithHistory* iTrack, const G4Track* track, bool inHistory, bool withAncestor);

  int giveMotherNeeded(int i) const {
    int theResult = 0;
    for (auto& p : idsave) {
      if (p.first == i) {
        theResult = p.second;
        break;
      }
    }
    return theResult;
  }

  bool trackExists(unsigned int i) const {
    bool flag = false;
    for (auto& ptr : m_trackContainer) {
      if (ptr->trackID() == i) {
        flag = true;
        break;
      }
    }
    return flag;
  }

  TrackWithHistory* getTrackByID(unsigned int trackID, bool strict = false) const {
    TrackWithHistory* track = nullptr;
    for (auto& ptr : m_trackContainer) {
      if (ptr->trackID() == trackID) {
        track = ptr;
        break;
      }
    }
    if (nullptr == track && strict) {
      ReportException(trackID);
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
  void ReportException(unsigned int id) const;

  // to restore the pre-LHC Transport GenParticle id link to a SimTrack
  void resetGenID();

  // ---------- member data --------------------------------

  int m_nVertices{0};
  unsigned int lastTrack{0};
  unsigned int lastHist{0};

  const edm::LHCTransportLinkContainer* theLHCTlink{nullptr};

  VertexMap m_vertexMap;
  std::vector<std::pair<int, int> > idsave;
  std::vector<std::pair<int, int> > ancestorList;
  std::vector<std::pair<int, math::XYZVectorD> > m_endPoints;
  std::vector<TrackWithHistory*> m_trackContainer;
};

class trkIDLess {
public:
  bool operator()(TrackWithHistory* trk1, TrackWithHistory* trk2) const { return (trk1->trackID() < trk2->trackID()); }
};

#endif
