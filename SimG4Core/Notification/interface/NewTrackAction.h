#ifndef SimG4Core_NewTrackAction_H
#define SimG4Core_NewTrackAction_H

class G4Track;
class TrackInformation;

/** SimG4Core Action for new G4tracks.
 *  This action is called each time a new G4Track is formed.
 *  Since formation (i.e. filling of data members) is done 
 *  gradually, the best moment to call NewTrackAction is not very clear.
 *  Currently done from StackingAction...
 */

class NewTrackAction {
public:
  NewTrackAction();
  void primary(const G4Track* aSecondary) const;
  void primary(G4Track* aSecondary) const;
  void secondary(const G4Track* aSecondary, const G4Track& mother, int) const;
  void secondary(G4Track* aSecondary, const G4Track& mother, int) const;

private:
  void addUserInfoToPrimary(G4Track* aTrack) const;
  void addUserInfoToSecondary(G4Track* aTrack, const TrackInformation& motherInfo, int) const;
};

#endif
