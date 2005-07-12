#ifndef CoreSimTrack_H
#define CoreSimTrack_H
 
#include <CLHEP/Vector/LorentzVector.h>
 
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
    //const HepParticleData * particleInfo() const;
    /// four momentum
    HepLorentzVector momentum() const 
    { return theMomentum; }
    /// particle type (HEP PDT convension)
    int type() const { return thePID;}
    /// charge
    //float charge() const;
private:
    int thePID;
    HepLorentzVector theMomentum;
};

#include <iosfwd>
std::ostream & operator <<(std::ostream & o , const CoreSimTrack & t);

#endif 
