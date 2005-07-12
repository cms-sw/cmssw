#include "SimDataFormats/Track/interface/EmbdSimTrack.h"

EmbdSimTrack::EmbdSimTrack() {}
 
EmbdSimTrack::EmbdSimTrack(int ipart, const HepLorentzVector & p) :
    Core(ipart, p), ivert(-1), igenpart(-1) {}
 
EmbdSimTrack::EmbdSimTrack(int ipart, const HepLorentzVector & p, int iv, int ig) :
    Core(ipart, p), ivert(iv), igenpart(ig) {}
 
EmbdSimTrack::EmbdSimTrack(const CoreSimTrack & t, int iv, int ig) :
    Core(t), ivert(iv), igenpart(ig) {}
 
std::ostream & operator <<(std::ostream & o , const EmbdSimTrack & t) 
{
    return o << (EmbdSimTrack::Core)(t) << ", "
	     << t.vertIndex() << ", "
	     << t.genpartIndex();
}
