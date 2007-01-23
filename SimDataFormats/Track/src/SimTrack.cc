#include "SimDataFormats/Track/interface/SimTrack.h"

SimTrack::SimTrack() {}
 
SimTrack::SimTrack(int ipart, const HepLorentzVector & p) :
    Core(ipart, p), ivert(-1), igenpart(-1),tkposition(0.),tkmomentum(0.) {}
 
SimTrack::SimTrack(int ipart, const HepLorentzVector & p, int iv, int ig) :
    Core(ipart, p), ivert(iv), igenpart(ig),tkposition(0.),tkmomentum(0.)  {}

SimTrack::SimTrack(int ipart, const HepLorentzVector & p, int iv, int ig,
		   const Hep3Vector &  tkp, const HepLorentzVector & tkm) :
    Core(ipart, p), ivert(iv), igenpart(ig),tkposition(tkp),tkmomentum(tkm)  {}
 
SimTrack::SimTrack(const CoreSimTrack & t, int iv, int ig) :
    Core(t), ivert(iv), igenpart(ig) {}
 
std::ostream & operator <<(std::ostream & o , const SimTrack & t) 
{
    return o << (SimTrack::Core)(t) << ", "
	     << t.vertIndex() << ", "
	     << t.genpartIndex();
}
