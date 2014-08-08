#ifndef CoreSimTrack_H
#define CoreSimTrack_H
 
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
  
class CoreSimTrack 
{ 
public:

    /// constructors
    CoreSimTrack() : tId(0), thePID(0) {}
    CoreSimTrack( int ipart, const math::XYZTLorentzVectorD& p ) :
      tId(0), thePID(ipart), theMomentum(p) {}

    CoreSimTrack( int ipart, math::XYZVectorD& ip, double ie ) :
      tId(0), thePID(ipart)
    { theMomentum.SetXYZT( ip.x(), ip.y(), ip.z(), ie ) ; }

    const math::XYZTLorentzVectorD& momentum() const { return theMomentum; }

    /// particle type (HEP PDT convension)
    int type() const { return thePID;}

    /// charge
    float charge() const;

    void setEventId(EncodedEventId e) {eId=e;}
    EncodedEventId eventId() const {return eId;}

    void setTrackId(unsigned int t) {tId=t;}
    unsigned int trackId() const {return tId;}

private:

    EncodedEventId eId;
    unsigned int tId;
    int thePID;
    math::XYZTLorentzVectorD theMomentum ;
};

#include <iosfwd>
std::ostream & operator <<(std::ostream & o , const CoreSimTrack & t);

#endif 
