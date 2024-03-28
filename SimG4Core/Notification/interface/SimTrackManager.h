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
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"

// forward declarations

class TmpSimEvent;
class TmpSimVertex;
class G4Track;

class SimTrackManager {
public:
  explicit SimTrackManager(TmpSimEvent*, int verbose);
  ~SimTrackManager();

  const std::vector<TrackWithHistory*>* trackContainer() const { return m_trackContainer; }

  void storeTracks();

  void reset();

  void addTrack(bool inHistory);
  //  void addTrack(bool inHistory, bool withAncestor);

  int giveMotherNeeded(int) const { return m_currTrackInfo->mcTruthID(); }

  bool trackExists(unsigned int) const { return true; }

  TrackWithHistory* getTrackByID(unsigned int, bool) const { return m_currHistory; }

  void setLHCTransportLink(const edm::LHCTransportLinkContainer* thisLHCTlink) { theLHCTlink = thisLHCTlink; }

  void initialisePrimaries(const G4Event*);

  TrackWithHistory* getTrackWithHistory(const G4Track*);

  const G4Track* getCurrentTrack() const { return m_currTrack; }

  // stop default
  SimTrackManager(const SimTrackManager&) = delete;
  const SimTrackManager& operator=(const SimTrackManager&) = delete;

private:
  //  void saveTrackAndItsBranch(TrackWithHistory*);
  int findOrAddVertex(math::XYZVectorD& pos, double& time, int i1, int i2);

  // ---------- member data --------------------------------

  int m_nPrimary{0};
  int m_nTracks{0};
  int m_nVertices{0};
  int m_nPrimVertices{0};
  int m_Verbose;

  TmpSimEvent* m_simEvent{nullptr};
  TmpSimVertex* m_simVertex{nullptr};
  const G4Track* m_currTrack{nullptr};
  TrackWithHistory* m_currHistory{nullptr};
  TrackInformation* m_currTrackInfo{nullptr};
  const edm::LHCTransportLinkContainer* theLHCTlink{nullptr};
  std::vector<TrackWithHistory*>* m_trackContainer;
  std::vector<TmpSimVertex*>* m_g4vertices;
};

#endif
