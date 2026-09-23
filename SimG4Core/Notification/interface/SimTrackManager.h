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
#include <unordered_map>
#include <vector>

// user include files
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"

// forward declarations

class TmpSimEvent;
class G4Track;

class SimTrackManager {
public:
  class StrictWeakOrdering {
  public:
    bool operator()(TrackWithHistory*& p, const int& i) const { return p->trackID() < i; }
  };

  typedef std::pair<int, math::XYZVectorD> VertexPosition;
  typedef std::vector<std::pair<int, math::XYZVectorD> > VertexPositionVector;
  typedef std::map<int, VertexPositionVector> VertexMap;

  explicit SimTrackManager(TmpSimEvent*, int);
  ~SimTrackManager();

  const std::vector<TrackWithHistory*>* trackContainer() const { return &m_trackContainer; }

  void storeTracks();
  void reset();
  void deleteTracks();
  void cleanTracksWithHistory();

  void addTrack(TrackWithHistory* iTrack, const G4Track* track, bool inHistory, bool withAncestor);

  int giveMotherNeeded(int i) const {
    int theResult = 0;
    for (auto const& p : idsave) {
      if (p.first == i) {
        theResult = p.second;
        break;
      }
    }
    return theResult;
  }

  bool trackExists(int i) const {
    bool flag = false;
    for (auto const& ptr : m_trackContainer) {
      if (ptr->trackID() == i) {
        flag = true;
        break;
      }
    }
    return flag;
  }

  TrackWithHistory* getTrackByID(int trackID, bool strict = false) const {
    TrackWithHistory* track = nullptr;
    for (auto const& ptr : m_trackContainer) {
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

  // When true, the production vertex of a stored track whose immediate parent was
  // *not* persisted (e.g. a sub-PersistencyEmin intermediate) is reattached to the
  // nearest stored ancestor instead of being left parent-less (parentIndex = -1).
  // This keeps the SimTrack/SimVertex history connected to the generator at a high
  // PersistencyEmin, at the cost of collapsing the dropped intermediate nodes.
  void setReconnectDroppedAncestors(bool v) { m_reconnectDroppedAncestors = v; }

  // Pure, unit-testable core of buildDroppedAncestorRedirect(): for each stored
  // track whose immediate parent is not in the stored set, walk the full
  // trackID -> parentID map (parentOfAll) up to the nearest stored ancestor and
  // record trackID -> ancestorTrackID. Primaries (parentID <= 0) and tracks whose
  // immediate parent is already stored get no entry; walks are capped to guard
  // against a malformed parent loop.
  static std::unordered_map<int, int> computeDroppedAncestorRedirect(
      const std::vector<std::pair<int, int> >& storedTracks, const std::unordered_map<int, int>& parentOfAll);

  // stop default
  SimTrackManager(const SimTrackManager&) = delete;
  const SimTrackManager& operator=(const SimTrackManager&) = delete;

private:
  void saveTrackAndItsBranch(TrackWithHistory*);
  int getOrCreateVertex(TrackWithHistory*, int);
  // For each stored track whose immediate parent was dropped, precompute the
  // trackID of its nearest stored ancestor (walking the full idsave parent map).
  // Called from storeTracks() before idsave is consumed, only when enabled.
  void buildDroppedAncestorRedirect();
  void cleanVertexMap();
  void reallyStoreTracks();
  void fillMotherList();
  int idSavedTrack(int) const;
  void ReportException(unsigned int id) const;

  // to restore the pre-LHC Transport GenParticle id link to a SimTrack
  void resetGenID();

  // ---------- member data --------------------------------

  int m_nVertices{0};
  unsigned int lastTrack{0};
  unsigned int lastHist{0};

  TmpSimEvent* m_simEvent;
  const edm::LHCTransportLinkContainer* theLHCTlink{nullptr};

  VertexMap m_vertexMap;
  std::vector<std::pair<int, int> > idsave;
  std::vector<std::pair<int, int> > ancestorList;
  std::vector<std::pair<int, math::XYZVectorD> > m_endPoints;
  std::vector<TrackWithHistory*> m_trackContainer;

  bool m_reconnectDroppedAncestors{false};
  // Complete trackID -> immediate parentID map for every simulated track of the
  // event (filled in addTrack when enabled). idsave is consumed per primary by
  // fillMotherList(), so this independent copy is needed to walk the full parent
  // chain at storeTracks() time. Cleared per event in reset().
  std::unordered_map<int, int> m_parentOfAll;
  // stored-track trackID -> nearest stored-ancestor trackID, only for tracks whose
  // immediate parent was dropped (filled by buildDroppedAncestorRedirect()).
  std::unordered_map<int, int> m_droppedParentRedirect;
};

class trkIDLess {
public:
  bool operator()(TrackWithHistory* trk1, TrackWithHistory* trk2) const { return (trk1->trackID() < trk2->trackID()); }
};

#endif
