#ifndef SimTrack_H
#define SimTrack_H

#include "SimDataFormats/Track/interface/CoreSimTrack.h"

class SimTrack : public CoreSimTrack
{

public:

  typedef CoreSimTrack Core;

  /// constructor
  SimTrack();
  SimTrack(int ipart, const HepLorentzVector& p);
  SimTrack(int ipart, const math::XYZTLorentzVectorD& p) ;

  /// full constructor (pdg type, momentum, time,
  /// index of parent vertex in final vector
  /// index of corresponding gen part in final vector)
  SimTrack(int ipart, const HepLorentzVector& p, int iv, int ig);
  SimTrack(int ipart, const math::XYZTLorentzVectorD& p, int iv, int ig);

  SimTrack(int ipart, const HepLorentzVector & p, int iv, int ig, 
	   const Hep3Vector & tkp, const HepLorentzVector & tkm);
  SimTrack(int ipart, const math::XYZTLorentzVectorD& p, int iv, int ig, 
	   const math::XYZVectorD& tkp, const math::XYZTLorentzVectorD& tkm);

  /// constructor from transient
  SimTrack(const CoreSimTrack& t, int iv, int ig);

  /// index of the vertex in the Event container (-1 if no vertex)
  int vertIndex() const { return ivert;}
  bool  noVertex() const { return ivert==-1;}

  /// index of the corresponding Generator particle in the Event container (-1 if no Genpart)
  int genpartIndex() const { return igenpart;}
  bool  noGenpart() const { return igenpart==-1;}

  Hep3Vector trackerSurfacePosition() 
  { return Hep3Vector( tkposition.x(), tkposition.y(), tkposition.z() ) ; }
  const math::XYZVectorD& trackerSurfacePosition() const { return tkposition; }

  HepLorentzVector trackerSurfaceMomentum()  
  { return HepLorentzVector( tkmomentum.x(), tkmomentum.y(), tkmomentum.z(), tkmomentum.e() ); }
  const math::XYZTLorentzVectorD& trackerSurfaceMomentum() const { return tkmomentum; }
  
  inline void setTkPosition(const Hep3Vector& pos) { tkposition.SetXYZ( pos.x(), pos.y(), pos.z() ) ; }
  inline void setTkPosition(const math::XYZVectorD& pos) { tkposition = pos; }
  inline void setTkMomentum(const HepLorentzVector& mom) 
  { tkmomentum.SetXYZT( mom.x(), mom.y(), mom.z(), mom.e() ) ; }
  inline void setTkMomentum(const math::XYZTLorentzVectorD& mom) { tkmomentum = mom; }

private: 

    int ivert;
    int igenpart;
    // Hep3Vector tkposition;
    // HepLorentzVector tkmomentum;
    math::XYZVectorD tkposition ;
    math::XYZTLorentzVectorD tkmomentum ;

};

#include <iosfwd>
std::ostream & operator <<(std::ostream & o , const SimTrack& t);

#endif
