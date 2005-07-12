#ifndef SimG4Core_TrackInformationExtractor_H
#define SimG4Core_TrackInformationExtractor_H

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"

class G4Track;

/** Provides safe access to the TrackInformation part of a G4Track.
 *  If the G4Track pointer/reference is const, a const reference to
 *  the TrackInformation is returned, else a non-const reference
 *  is  returned.
 *  The TrackInformationExtractor checks for the existance of 
 *  TrackInformation and for the validity of the dynamic_cast;
 *  It will throw an excepton if it is not possible to return 
 *  TrackInformation, but it will do it's best to satisfy the request
 *  (in some cases it may even create the TrackInformation on-the-fly).
 */

class TrackInformationExtractor 
{
public:
    /** for a const G4Track pointer/reference a const TrackInformation&
     *  is returned.
     */
    const TrackInformation & operator()(const G4Track & gtk) const;
    const TrackInformation & operator()(const G4Track * gtk) const { return operator()(*gtk); }
    /** for a non-const G4Track pointer/reference the TrackInformation&
     *  is also non-const.
     */
    TrackInformation & operator()(G4Track & gtk) const;
    TrackInformation & operator()(G4Track * gtk) const { return operator()(*gtk); }
private:
    void missing(const G4Track & gtk) const 
    { throw SimG4Exception("TrackInformationExtractor: G4Track has no TrackInformation"); }    
    void wrongType() const
    { throw SimG4Exception("User information in G4Track is not of TrackInformation type"); } 
};

#endif
