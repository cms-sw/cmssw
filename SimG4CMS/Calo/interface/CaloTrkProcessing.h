#ifdef messageimpl
#ifndef CaloTrkProcessing_H
#define CaloTrkProcessing_H

/* #include "Utilities/Notification/interface/Observer.h" */
/* #include "Utilities/Notification/interface/LazyObserver.h" */
/* #include "Utilities/UI/interface/Verbosity.h" */
#include "G4VTouchable.hh"
#include <map>
#include <iostream>

class BeginOfRun;
class BeginOfEvent;
class EndOfTrack;
class G4Step;

class CaloTrkProcessing : private Observer<const BeginOfRun *>, 
			  private Observer<const BeginOfEvent *>, 
			  private Observer<const EndOfTrack *>, 
			  private Observer<const G4Step *> {

public:

  CaloTrkProcessing();

private:

  void upDate(const BeginOfRun * run);
  void upDate(const BeginOfEvent * evt);
  void upDate(const EndOfTrack * trk);
  void upDate(const G4Step *);

  // Utilities to get detector levels during a step
  int      detLevels(const G4VTouchable*) const;
  G4String detName(const G4VTouchable*, int, int) const;
  void detectorLevel(const G4VTouchable*, int&, int*, G4String*) const;

  double rinCalo, zinCalo;
  bool   testBeam;
  int    lastTrackID;

  //  static UserVerbosity cout;

};

#endif
#endif








