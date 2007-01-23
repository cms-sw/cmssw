#ifndef SimTrack_H
#define SimTrack_H

#include "SimDataFormats/Track/interface/CoreSimTrack.h"

class SimTrack : public CoreSimTrack
{
public:
    typedef CoreSimTrack Core;
    /// constructor
    SimTrack();
    SimTrack(int ipart, const HepLorentzVector & p);
    /// full constructor (pdg type, momentum, time,
    /// index of parent vertex in final vector
    /// index of corresponding gen part in final vector)
    SimTrack(int ipart, const HepLorentzVector & p, int iv, int ig);
    SimTrack(int ipart, const HepLorentzVector & p, int iv, int ig, 
	     const Hep3Vector & tkp, const HepLorentzVector & tkm);
    /// constructor from transient
    SimTrack(const CoreSimTrack & t, int iv, int ig);
    /// index of the vertex in the Event container (-1 if no vertex)
    int vertIndex() const { return ivert;}
    bool  noVertex() const { return ivert==-1;}
    /// index of the corresponding Generator particle in the Event container (-1 if no Genpart)
    int genpartIndex() const { return igenpart;}
    bool  noGenpart() const { return igenpart==-1;}
    //
    Hep3Vector trackerSurfacePosition() const { return tkposition; }
    HepLorentzVector trackerSurfaceMomentum() const { return tkmomentum; }
private: 
    int ivert;
    int igenpart;
    Hep3Vector tkposition;
    HepLorentzVector tkmomentum;
};

#include <iosfwd>
std::ostream & operator <<(std::ostream & o , const SimTrack& t);

#endif
