#ifndef TrackPropagation_Geant4eSteppingAction_h
#define TrackPropagation_Geant4eSteppingAction_h

#include "G4UserSteppingAction.hh"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"


/** A G4 User stepping action used to calculate the total track. The method
    G4UserSteppingAction::UserSteppingAction(const G4Step*) should be 
    automatically called by G4eManager at each step. 

 */
class Geant4eSteppingAction GCC11_FINAL : public G4UserSteppingAction {
 public:
  Geant4eSteppingAction():theTrackLength(0) {}
  virtual ~Geant4eSteppingAction() {}

  /** Retrieve the length that the track has accumulated since the last call
      to reset()
  */
  double trackLength() const {return theTrackLength;}

  /** Resets to 0 the counter on the track length. Should be called at the 
      beginning of any extrapolation.
  */
  void reset() {theTrackLength = 0;}

  /** This method is automatically called by G4eManager at each step. The step
      length is then added to the stored value of the track length.     
   */
  virtual void UserSteppingAction(const G4Step* step);
  
 protected:
  double theTrackLength;
};



#endif
