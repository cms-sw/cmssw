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
    /// constructor from transient
    SimTrack(const CoreSimTrack & t, int iv, int ig);
    /// index of the vertex in the Event container (-1 if no vertex)
    int vertIndex() const { return ivert;}
    bool  noVertex() const { return ivert==-1;}
    /// index of the corresponding Generator particle in the Event container (-1 if no Genpart)
    int genpartIndex() const { return igenpart;}
    bool  noGenpart() const { return igenpart==-1;}
private: 
    int ivert;
    int igenpart;
};

#include <iosfwd>
std::ostream & operator <<(std::ostream & o , const SimTrack& t);

#endif
