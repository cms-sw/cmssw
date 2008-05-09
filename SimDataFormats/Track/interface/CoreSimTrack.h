#ifndef CoreSimTrack_H
#define CoreSimTrack_H
 
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

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
    CoreSimTrack( int ipart, const math::XYZTLorentzVectorD& p ) :
       thePID(ipart), theMomentum(p) {}

    CoreSimTrack( int ipart, math::XYZVectorD& ip, double ie ) :
       thePID(ipart)
    { theMomentum.SetXYZT( ip.x(), ip.y(), ip.z(), ie ) ; }

    /// particle info...
    //    const HepPDT::ParticleData * particleInfo() const;

    /// four momentum
//    HepLorentzVector momentum() { return HepLorentzVector( theMomentum.px(),
//                                                           theMomentum.py(),
//							   theMomentum.pz(),
//						           theMomentum.e()  ) ; }
    const math::XYZTLorentzVectorD& momentum() const { return theMomentum; }
    // math::XYZTLorentzVectorD& momentum() { return theMomentum; }

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
    math::XYZTLorentzVectorD theMomentum ;
};

#include <iosfwd>
std::ostream & operator <<(std::ostream & o , const CoreSimTrack & t);

#endif 
