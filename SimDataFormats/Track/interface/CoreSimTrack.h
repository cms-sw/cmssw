#ifndef CoreSimTrack_H
#define CoreSimTrack_H
 
#include <CLHEP/Vector/LorentzVector.h>
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
  
#include <cmath>
 
//class HepParticleData;
 
/**  a generic Simulated Track
 */
class CoreSimTrack 
{ 
public:
    /// constructors
    CoreSimTrack() {}
    CoreSimTrack(int ipart, const HepLorentzVector & p) : thePID(ipart), theMomentum(p) {}  
    CoreSimTrack(int ipart, const Hep3Vector & ip, double ie) : thePID(ipart) 
    { theMomentum[0] = ip.x(); theMomentum[1] = ip.y(); 
      theMomentum[2] = ip.z(); theMomentum[3] = ie; }
    /// particle info...
    //    const HepPDT::ParticleData * particleInfo() const;
    /// four momentum
    const HepLorentzVector & momentum() const 
    { return theMomentum; }
    /// particle type (HEP PDT convension)
    int type() const { return thePID;}
    /// charge
    float charge() const;
    void setEventId(EncodedEventId e) {eId=e;}
    EncodedEventId eventId() const {return eId;}
    void setTrackId(unsigned int t) {tId=t;}
    unsigned int trackId() const {return tId;}
private:
    int chargeValue(const int&)const;
    EncodedEventId eId;
    unsigned int tId;
    int thePID;
    HepLorentzVector theMomentum;
};

#include <iosfwd>
std::ostream & operator <<(std::ostream & o , const CoreSimTrack & t);

#endif 
