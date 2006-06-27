#include "SimDataFormats/Track/interface/CoreSimTrack.h"
 
#include "HepPDT/defs.h"
#include "HepPDT/TableBuilder.hh"

#include <fstream>

const HepPDT::ParticleData * CoreSimTrack::particleInfo() const 
{ 
    const char * in1 = "data/pdt.table";
    std::ifstream pdf1(in1);
    if (!pdf1) 
    { std::cout << " input file not found " << std::endl; return 0; }
    HepPDT::ParticleDataTable pdt("PDT table");
    HepPDT::TableBuilder tb(pdt);
    if (!HepPDT::addPDGParticles(pdf1, tb)) 
    { std::cout << "error reading PDG file " << std::endl; return 0; }
    HepPDT::ParticleData * pd = pdt.particle(HepPDT::ParticleID(type()));
    return pd; 
}
 
float CoreSimTrack::charge() const 
{ return particleInfo()->charge(); }
 
std::ostream & operator <<(std::ostream & o , const CoreSimTrack& t) 
{
    o << t.type() << ", ";
    o << t.momentum();
    return o;
}
