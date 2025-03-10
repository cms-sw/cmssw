#ifndef SimG4Core_CMSG4TrackInterface_h
#define SimG4Core_CMSG4TrackInterface_h 1

// 
// Package:     Application
// Class  :     SimTrackInterface
//
// 10.03.2025   V.Ivantchenko
// 
// An interface between Geant4 and CMSSW

#include "G4ThreadLocalSingleton.hh"

class G4Track;

class CMSG4TrackInterface {

friend class G4ThreadLocalSingleton<CMSG4TrackInterface>;

public:

  static CMSG4TrackInterface* instance();

  ~CMSG4TrackInterface();

  const G4Track* getCurrentTrack() { return track_; }

  void setCurrentTrack(const G4Track* p) { track_ = p; }

  int getThreadID() { return threadID_; }

  void setThreadID(int n) { threadID_ = n; }

  CMSG4TrackInterface(CMSG4TrackInterface&) = delete;
  CMSG4TrackInterface& operator=(const CMSG4TrackInterface& right) = delete;

private:

  CMSG4TrackInterface();
  
  static G4ThreadLocal CMSG4TrackInterface* interface_;

  const G4Track* track_{nullptr};
  int threadID_{0};
};

#endif
