#include "SimDataFormats/Track/interface/CoreSimTrack.h"
 
// #include "GeneratorInterface/HepPDT/interface/HepPDTable.h"
// #include "GeneratorInterface/HepPDT/interface/HepParticleData.h"
 
// const HepParticleData * CoreSimTrack::particleInfo() const 
// { return HepPDT::theTable().getParticleData(type()); }
 
//float CoreSimTrack::charge() const { return particleInfo()->charge(); }
 
std::ostream & operator <<(std::ostream & o , const CoreSimTrack& t) 
{
    o << t.type() << ", ";
    o << t.momentum();
    return o;
}
