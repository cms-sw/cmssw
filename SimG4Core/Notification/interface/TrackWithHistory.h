#ifndef SimG4Core_TrackWithHistory_H
#define SimG4Core_TrackWithHistory_H 

#include "G4Track.hh"

class G4VProcess;
class G4TrackToParticleID;
/** The part of the information about a SimTrack that we need from
 *  a G4Track
 */

class TrackWithHistory 
{
public:
    /** The constructor is called at PreUserTrackingAction time, 
     *  when some of the information is not available yet.
     */
    TrackWithHistory(const G4Track * g4track);
    ~TrackWithHistory() {}
    void save()					 { saved_ = true; }
    unsigned int trackID() const                 { return trackID_; }
    int particleID() const                       { return particleID_; }
    int parentID() const                         { return parentID_; }
    int genParticleID() const                    { return genParticleID_; }
    const Hep3Vector& momentum() const           { return momentum_; }
    double totalEnergy() const                   { return totalEnergy_; }
    const Hep3Vector& vertexPosition() const     { return vertexPosition_; }
    double globalTime() const                    { return globalTime_; }
    double localTime() const                     { return localTime_; }
    double properTime() const                    { return properTime_; }
    const G4VProcess * creatorProcess() const    { return creatorProcess_; }
    double weight() const                        { return weight_; }
    void setTrackID(int i)  			 { trackID_ = i; }
    void setParentID(int i)			 { parentID_ = i; }
    bool storeTrack() const                      { return storeTrack_; }
    bool saved() const                           { return saved_; }
    /** Internal consistency check (optional).
     *  Method called at PostUserTrackingAction time, to check
     *  if the information is consistent with that provided
     *  to the constructor.
     */
    void checkAtEnd(const G4Track *);
private:
    unsigned int trackID_;
    int particleID_;
    int parentID_;
    int genParticleID_;
    Hep3Vector momentum_;
    double totalEnergy_;
    Hep3Vector vertexPosition_;
    double globalTime_;
    double localTime_;
    double properTime_;
    const G4VProcess * creatorProcess_;
    double weight_;
    bool storeTrack_;
    bool saved_;
    static G4TrackToParticleID*  theG4TrackToParticleID;
    int extractGenID(const G4Track * gt) const;
};

#endif
