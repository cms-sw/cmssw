#include "SimDataFormats/Track/interface/SimTrack.h"

SimTrack::SimTrack() {}
 
SimTrack::SimTrack(int ipart, const HepLorentzVector & p) :
    Core(ipart, p), ivert(-1), igenpart(-1) {}
 
SimTrack::SimTrack(int ipart, const HepLorentzVector & p, int iv, int ig) :
    Core(ipart, p), ivert(iv), igenpart(ig) {}
 
SimTrack::SimTrack(const CoreSimTrack & t, int iv, int ig) :
    Core(t), ivert(iv), igenpart(ig) {}
 
std::ostream & operator <<(std::ostream & o , const SimTrack & t) 
{
    return o << (SimTrack::Core)(t) << ", "
	     << t.vertIndex() << ", "
	     << t.genpartIndex();
}
