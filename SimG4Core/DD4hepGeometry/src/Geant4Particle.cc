//==========================================================================
//  AIDA Detector description implementation 
//--------------------------------------------------------------------------
// Copyright (C) Organisation europeenne pour la Recherche nucleaire (CERN)
// All rights reserved.
//
// For the licensing terms see $DD4hepINSTALL/LICENSE.
// For the list of contributors see $DD4hepINSTALL/doc/CREDITS.
//
// Author     : M.Frank
//
//==========================================================================

// Framework include files
#include "DD4hep/Printout.h"
#include "DD4hep/Primitives.h"
#include "DD4hep/InstanceCount.h"
//FIXME: #include "DDG4/Geant4Particle.h"
#include "SimG4Core/DD4hepGeometry/interface/Geant4Particle.h"
#include "TDatabasePDG.h"
#include "TParticlePDG.h"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4VProcess.hh"
#include "G4ChargedGeantino.hh"
#include "G4Geantino.hh"

#include <sstream>
#include <iostream>
#include <regex.h>

using namespace dd4hep;
using namespace dd4hep::sim;
typedef detail::ReferenceBitMask<int> PropertyMask;

/// Default destructor
ParticleExtension::~ParticleExtension() {
}

/// Default constructor
Geant4Particle::Geant4Particle() : ref(1)
{
  InstanceCount::increment(this);
}

/// Constructor with ID initialization
Geant4Particle::Geant4Particle(int part_id) : ref(1), id(part_id), originalG4ID(part_id)
{
  InstanceCount::increment(this);
}

/// Default destructor
Geant4Particle::~Geant4Particle()  {
  InstanceCount::decrement(this);
  //::printf("************ Delete Geant4Particle[%p]: ID:%d pdgID %d ref:%d\n",(void*)this,id,pdgID,ref);
}

void Geant4Particle::release()  {
  //::printf("************ Release Geant4Particle[%p]: ID:%d pdgID %d ref:%d\n",(void*)this,id,pdgID,ref-1);
  if ( --ref <= 0 )  {
    delete this;
  }
}

/// Assignment operator
Geant4Particle& Geant4Particle::get_data(Geant4Particle& c)   {
  if ( this != &c )  {
    id = c.id;
    originalG4ID = c.originalG4ID;
    g4Parent    = c.g4Parent;
    reason      = c.reason;
    mask        = c.mask;
    status      = c.status;
    genStatus   = c.genStatus;
    charge      = c.charge;
    steps       = c.steps;
    secondaries = c.secondaries;
    pdgID       = c.pdgID;
    vsx         = c.vsx;
    vsy         = c.vsy;
    vsz         = c.vsz;
    vex         = c.vex;
    vey         = c.vey;
    vez         = c.vez;
    psx         = c.psx;
    psy         = c.psy;
    psz         = c.psz;
    pex         = c.pex;
    pey         = c.pey;
    pez         = c.pez;
    mass        = c.mass;
    time        = c.time;
    properTime  = c.properTime;
    process     = c.process;
    //definition  = c.definition;
    daughters   = c.daughters;
    parents     = c.parents;
#if __cplusplus >= 201103L
    extension.swap(c.extension);
#else
    extension   = c.extension;
#endif
    //DD4hep_ptr<ParticleExtension>(c.extension.release());
  }
  return *this;
}

/// Remove daughter from set
void Geant4Particle::removeDaughter(int id_daughter)  {
  std::set<int>::iterator j = daughters.find(id_daughter);
  if ( j != daughters.end() ) daughters.erase(j);
}

/// Access the Geant4 particle definition object (expensive!)
const G4ParticleDefinition* Geant4ParticleHandle::definition() const   {
  G4ParticleTable*      tab = G4ParticleTable::GetParticleTable();
  G4ParticleDefinition* def = tab->FindParticle(particle->pdgID);
  if ( 0 == def && 0 == particle->pdgID )   {
    if ( 0 == particle->charge )
      return G4Geantino::Definition();
    return G4ChargedGeantino::Definition();
  }
  return def;
}

/// Access to the Geant4 particle name
std::string Geant4ParticleHandle::particleName() const   {
  const G4ParticleDefinition* def = definition();
  if ( def )   {
    //particle->definition = def;
    return def->GetParticleName();
  }
#if 0
  TDatabasePDG* db = TDatabasePDG::Instance();
  TParticlePDG* pdef = db->GetParticle(particle->pdgID);
  if ( pdef ) return pdef->GetName();
#endif
  char text[32];
  ::snprintf(text,sizeof(text),"PDG:%d",particle->pdgID);
  return text;
}

/// Access to the Geant4 particle type
std::string Geant4ParticleHandle::particleType() const   {
  const G4ParticleDefinition* def = definition();
  if ( def )   {
    //particle->definition = def;
    return def->GetParticleType();
  }
#if 0
  TDatabasePDG* db = TDatabasePDG::Instance();
  TParticlePDG* pdef = db->GetParticle(particle->pdgID);
  if ( pdef ) return pdef->ParticleClass();
#endif
  char text[32];
  ::snprintf(text,sizeof(text),"PDG:%d",particle->pdgID);
  return text;
}

/// Access Geant4 particle definitions by regular expression
std::vector<G4ParticleDefinition*> Geant4ParticleHandle::g4DefinitionsRegEx(const std::string& expression)   {
  std::vector<G4ParticleDefinition*> results;
  std::string exp = expression;   //'^'+expression+"$";
  G4ParticleTable* pt = G4ParticleTable::GetParticleTable();
  G4ParticleTable::G4PTblDicIterator* iter = pt->GetIterator();
  char msgbuf[128];
  regex_t reg;
  int ret = ::regcomp(&reg, exp.c_str(), 0);
  if (ret) {
    throw std::runtime_error(format("Geant4ParticleHandle", "REGEX: Failed to compile particle name %s", exp.c_str()));
  }
  results.clear();
  iter->reset();
  while ((*iter)()) {
    G4ParticleDefinition* p = iter->value();
    ret = ::regexec(&reg, p->GetParticleName().c_str(), 0, NULL, 0);
    if (!ret)
      results.push_back(p);
    else if (ret == REG_NOMATCH)
      continue;
    else {
      ::regerror(ret, &reg, msgbuf, sizeof(msgbuf));
      ::regfree(&reg);
      throw std::runtime_error(format("Geant4ParticleHandle", "REGEX: Failed to match particle name %s err=%s", exp.c_str(), msgbuf));
    }
  }
  ::regfree(&reg);
  return results;
}

/// Access Geant4 particle definitions by exact match
G4ParticleDefinition* Geant4ParticleHandle::g4DefinitionsExact(const std::string& expression)   {
  G4ParticleTable*      tab = G4ParticleTable::GetParticleTable();
  G4ParticleDefinition* def = tab->FindParticle(expression);
  return def;
}

/// Access to the creator process name
std::string Geant4ParticleHandle::processName() const   {
  if ( particle->process ) return particle->process->GetProcessName();
  else if ( particle->reason&G4PARTICLE_PRIMARY ) return "Primary";
  else if ( particle->status&G4PARTICLE_GEN_EMPTY ) return "Gen.Empty";
  else if ( particle->status&G4PARTICLE_GEN_STABLE ) return "Gen.Stable";
  else if ( particle->status&G4PARTICLE_GEN_DECAYED ) return "Gen.Decay";
  else if ( particle->status&G4PARTICLE_GEN_DOCUMENTATION ) return "Gen.DOC";
  else if ( particle->status&G4PARTICLE_GEN_BEAM ) return "Gen.Beam";
  else if ( particle->status&G4PARTICLE_GEN_OTHER ) return "Gen.Other";
  return "???";
}

/// Access to the creator process type name
std::string Geant4ParticleHandle::processTypeName() const   {
  if ( particle->process )   {
    return G4VProcess::GetProcessTypeName(particle->process->GetProcessType());
  }
  return "";
}

/// Offset all particle identifiers (id, parents, daughters) by a constant number
void Geant4ParticleHandle::offset(int off)  const   {
  std::set<int> temp;
  Geant4Particle* p = particle;
  p->id += off;

  temp = p->daughters;
  p->daughters.clear();
  for(std::set<int>::iterator i=temp.begin(); i != temp.end(); ++i)
    p->daughters.insert((*i)+off);

  temp = p->parents;
  p->parents.clear();
  for(std::set<int>::iterator i=temp.begin(); i != temp.end(); ++i)
    p->parents.insert((*i)+off);
}

/// Output type 1:+++ <tag>   10 def:0xde4eaa8 [gamma     ,   gamma] reason:      20 E:+1.017927e+03  \#Par:  1/4    \#Dau:  2
void Geant4ParticleHandle::dump1(int level, const std::string& src, const char* tag) const   {
  char text[256];
  Geant4ParticleHandle p(*this);
  text[0]=0;
  if ( p->parents.size() == 1 )
    ::snprintf(text,sizeof(text),"/%d",*(p->parents.begin()));
  else if ( p->parents.size() >  1 )   {
    text[0]='/';text[1]=0;
    for(std::set<int>::const_iterator i=p->parents.begin(); i!=p->parents.end(); ++i)
      ::snprintf(text+strlen(text),sizeof(text)-strlen(text),"%d ",*i);
  }
  printout((dd4hep::PrintLevel)level,src,
           "+++ %s %4d def [%-11s,%8s] reason:%8d E:%+.2e %3s #Dau:%3d #Par:%3d%-5s",
           tag, p->id,
           p.particleName().c_str(),
           p.particleType().c_str(),
           p->reason,
           p.energy(),
           p->g4Parent>0 ? "Sim" : "Gen",
           int(p->daughters.size()),
           int(p->parents.size()),text);
}

/// Output type 2:+++ <tag>   20 G4:   7 def:0xde4eaa8 [gamma     ,   gamma] reason:      20 E:+3.304035e+01 in record:YES  \#Par:  1/18   \#Dau:  0
void Geant4ParticleHandle::dump2(int level, const std::string& src, const char* tag, int g4id, bool inrec) const   {
  char text[32];
  Geant4ParticleHandle p(*this);
  if ( p->parents.size() == 0 ) text[0]=0;
  else if ( p->parents.size() == 1 ) ::snprintf(text,sizeof(text),"/%d",*(p->parents.begin()));
  else if ( p->parents.size() >  1 ) ::snprintf(text,sizeof(text),"/%d..",*(p->parents.begin()));
  printout((dd4hep::PrintLevel)level,src,
           "+++ %s %4d G4:%4d [%-12s,%8s] reason:%8d "
           "E:%+.2e in record:%s  #Par:%3d%-5s #Dau:%3d",
           tag, p->id, g4id,
           p.particleName().c_str(),
           p.particleType().c_str(),
           p->reason,
           p.energy(),
           yes_no(inrec),
           int(p->parents.size()),text,
           int(p->daughters.size()));
}

/// Output type 3:+++ <tag> ID:  0 e-           status:00000014 type:       11 Vertex:(+0.00e+00,+0.00e+00,+0.00e+00) [mm] time: +0.00e+00 [ns] \#Par:  0 \#Dau:  4
void Geant4ParticleHandle::dumpWithVertex(int level, const std::string& src, const char* tag) const  {
  char text[256];
  Geant4ParticleHandle p(*this);
  text[0]=0;
  if ( p->parents.size() == 1 )
    ::snprintf(text,sizeof(text),"/%d",*(p->parents.begin()));
  else if ( p->parents.size() >  1 )   {
    text[0]='/';text[1]=0;
    for(std::set<int>::const_iterator i=p->parents.begin(); i!=p->parents.end(); ++i)
      ::snprintf(text+strlen(text),sizeof(text)-strlen(text),"%d ",*i);
  }
  printout((dd4hep::PrintLevel)level,src,
           "+++ %s ID:%3d %-12s status:%08X PDG:%6d Vtx:(%+.2e,%+.2e,%+.2e)[mm] "
           "time: %+.2e [ns] #Dau:%3d #Par:%1d%-6s",
           tag,p->id,p.particleName().c_str(),p->status,p->pdgID,
           p->vsx/CLHEP::mm,p->vsy/CLHEP::mm,p->vsz/CLHEP::mm,p->time/CLHEP::ns,
           p->daughters.size(),
           p->parents.size(),
           text);
}


/// Output type 3:+++ <tag> ID:  0 e-           status:00000014 type:       11 Vertex:(+0.00e+00,+0.00e+00,+0.00e+00) [mm] time: +0.00e+00 [ns] \#Par:  0 \#Dau:  4
void Geant4ParticleHandle::dumpWithMomentum(int level, const std::string& src, const char* tag) const  {
  char text[256];
  Geant4ParticleHandle p(*this);
  text[0]=0;
  if ( p->parents.size() == 1 )
    ::snprintf(text,sizeof(text),"/%d",*(p->parents.begin()));
  else if ( p->parents.size() >  1 )   {
    text[0]='/';text[1]=0;
    for(std::set<int>::const_iterator i=p->parents.begin(); i!=p->parents.end(); ++i)
      ::snprintf(text+strlen(text),sizeof(text)-strlen(text),"%d ",*i);
  }
  printout((dd4hep::PrintLevel)level,src,
           "+++%s ID:%3d %-12s stat:%08X PDG:%6d Mom:(%+.2e,%+.2e,%+.2e)[MeV] "
           "time: %+.2e [ns] #Dau:%3d #Par:%1d%-6s",
           tag,p->id,p.particleName().c_str(),p->status,p->pdgID,
           p->psx/CLHEP::MeV,p->psy/CLHEP::MeV,p->psz/CLHEP::MeV,p->time/CLHEP::ns,
           int(p->daughters.size()),
           int(p->parents.size()),
           text);
}

/// Output type 3:+++ <tag> ID:  0 e-           status:00000014 type:       11 Vertex:(+0.00e+00,+0.00e+00,+0.00e+00) [mm] time: +0.00e+00 [ns] \#Par:  0 \#Dau:  4
void Geant4ParticleHandle::dumpWithMomentumAndVertex(int level, const std::string& src, const char* tag) const  {
  char text[256];
  Geant4ParticleHandle p(*this);
  text[0]=0;
  if ( p->parents.size() == 1 )
    ::snprintf(text,sizeof(text),"/%d",*(p->parents.begin()));
  else if ( p->parents.size() >  1 )   {
    text[0]='/';text[1]=0;
    for(std::set<int>::const_iterator i=p->parents.begin(); i!=p->parents.end(); ++i)
      ::snprintf(text+strlen(text),sizeof(text)-strlen(text),"%d ",*i);
  }
  printout((dd4hep::PrintLevel)level,src,
           "+++%s %3d %-12s stat:%08X PDG:%6d Mom:(%+.2e,%+.2e,%+.2e)[MeV] "
           "Vtx:(%+.2e,%+.2e,%+.2e)[mm] #Dau:%3d #Par:%1d%-6s",
           tag,p->id,p.particleName().c_str(),p->status,p->pdgID,
           p->psx/CLHEP::MeV,p->psy/CLHEP::MeV,p->psz/CLHEP::MeV,
           p->vsx/CLHEP::mm,p->vsy/CLHEP::mm,p->vsz/CLHEP::mm,
           int(p->daughters.size()),
           int(p->parents.size()),
           text);
}

void Geant4ParticleHandle::header4(int level, const std::string& src, const char* tag)  {
  printout((dd4hep::PrintLevel)level,src,
           "+++ %s %10s/%-7s %12s/%-10s %6s/%-6s %4s %4s "
           "%-4s %-3s %-3s %-10s "
           "%-5s %-6s %-3s %-3s  %-20s %s",
           tag,"ID", "G4-ID", "Part-Name","PDG", "Parent","G4-ID", "#Par","#Dau",
           "Prim","Sec",">E","Energy",
           "EMPTY","STAB","DEC","DOC", 
           "Process", "Processing Flags");
}

void Geant4ParticleHandle::dump4(int level, const std::string& src, const char* tag) const  {
  Geant4ParticleHandle p(*this);
  //char equiv[32];
  PropertyMask mask(p->reason);
  PropertyMask status(p->status);
  std::string proc_name = p.processName();
  std::string proc_type = p.processTypeName();
  std::string proc = '['+proc_name+(p->process ? "/" : "")+proc_type+']';
  int parent_id = p->parents.empty() ? -1 : *(p->parents.begin());

  //equiv[0] = 0;
  //if ( p->parents.end() == p->parents.find(p->g4Parent) )  {
  //  ::snprintf(equiv,sizeof(equiv),"/%d",p->g4Parent);
  //}
  std::stringstream str;
  str << "Parents: ";
  for( const auto i : p->parents )
    str << i << " ";
  str << " Daughters: ";
  for( const auto i : p->daughters )
    str << i << " ";
  printout((dd4hep::PrintLevel)level,src,
           "+++ %s ID:%7d/%-7d %12s/%-10d %6d/%-6d %4d %4d %-4s %-3s %-3s %+.3e  "
           "%-5s %-4s %-3s %-3s  %-20s %c%c%c%c -- %c%c%c%c%c%c%c%c%c  %s",
           tag,
           p->id,p->originalG4ID,
           p.particleName().c_str(),
           p->pdgID,
           parent_id,p->g4Parent,
           p.numParent(),
           int(p->daughters.size()),
           yes_no(mask.isSet(G4PARTICLE_PRIMARY)),
           yes_no(mask.isSet(G4PARTICLE_HAS_SECONDARIES)),
           yes_no(mask.isSet(G4PARTICLE_ABOVE_ENERGY_THRESHOLD)),
           p.energy(),
           //
           yes_no(mask.isSet(G4PARTICLE_CREATED_CALORIMETER_HIT)),
           yes_no(mask.isSet(G4PARTICLE_CREATED_TRACKER_HIT)),
           yes_no(mask.isSet(G4PARTICLE_KEEP_PROCESS)),
           mask.isSet(G4PARTICLE_KEEP_PARENT) ? "YES" : "",
           proc.c_str(),
           // 13 flags in total
           status.isSet(G4PARTICLE_GEN_EMPTY) ? 'E' : '.',               // 1
           status.isSet(G4PARTICLE_GEN_STABLE) ? 'S' : '.',
           status.isSet(G4PARTICLE_GEN_DECAYED) ? 'D' : '.',
           status.isSet(G4PARTICLE_GEN_DOCUMENTATION) ? 'd' : '.',
           status.isSet(G4PARTICLE_GEN_BEAM) ? 'B' : '.',                // 5
           status.isSet(G4PARTICLE_GEN_OTHER) ? 'o' : '.',
           status.isSet(G4PARTICLE_SIM_CREATED) ? 's' : '.',
           status.isSet(G4PARTICLE_SIM_BACKSCATTER) ? 'b' : '.',
           status.isSet(G4PARTICLE_SIM_PARENT_RADIATED) ? 'v' : '.',
           status.isSet(G4PARTICLE_SIM_DECAY_TRACKER) ? 't' : '.',       // 10
           status.isSet(G4PARTICLE_SIM_DECAY_CALO) ? 'c' : '.',
           status.isSet(G4PARTICLE_SIM_LEFT_DETECTOR) ? 'l' : '.',
           status.isSet(G4PARTICLE_SIM_STOPPED) ? 's' : '.',
           str.str().c_str()
           );
#if 0
  printout((dd4hep::PrintLevel)level,src,
           "+++ %s ID:%7d %12s %6d%-7s %7s %3s %5d %3s %+.3e  %-4s %-7s %-3s %-3s %2d  [%s%s%s] %c%c%c%c -- %c%c%c%c%c%c%c",
           tag,
           p->id,
           p.particleName().c_str(),
           parent_id,equiv,
           yes_no(mask.isSet(G4PARTICLE_PRIMARY)),
           yes_no(mask.isSet(G4PARTICLE_HAS_SECONDARIES)),
           int(p->daughters.size()),
           yes_no(mask.isSet(G4PARTICLE_ABOVE_ENERGY_THRESHOLD)),
           p.energy(),
           yes_no(mask.isSet(G4PARTICLE_CREATED_CALORIMETER_HIT)),
           yes_no(mask.isSet(G4PARTICLE_CREATED_TRACKER_HIT)),
           yes_no(mask.isSet(G4PARTICLE_KEEP_PROCESS)),
           mask.isSet(G4PARTICLE_KEEP_PARENT) ? "YES" : "",
           p.numParent(),
           proc_name.c_str(),
           p->process ? "/" : "",
           proc_type.c_str(),
           status.isSet(G4PARTICLE_GEN_EMPTY) ? 'E' : '.',
           status.isSet(G4PARTICLE_GEN_STABLE) ? 'S' : '.',
           status.isSet(G4PARTICLE_GEN_DECAYED) ? 'D' : '.',
           status.isSet(G4PARTICLE_GEN_DOCUMENTATION) ? 'd' : '.',
           status.isSet(G4PARTICLE_GEN_BEAM) ? 'B' : '.',
           status.isSet(G4PARTICLE_GEN_OTHER) ? 'o' : '.',

           status.isSet(G4PARTICLE_SIM_CREATED) ? 's' : '.',
           status.isSet(G4PARTICLE_SIM_BACKSCATTER) ? 'b' : '.',
           status.isSet(G4PARTICLE_SIM_PARENT_RADIATED) ? 'v' : '.',
           status.isSet(G4PARTICLE_SIM_DECAY_TRACKER) ? 't' : '.',
           status.isSet(G4PARTICLE_SIM_DECAY_CALO) ? 'c' : '.',
           status.isSet(G4PARTICLE_SIM_LEFT_DETECTOR) ? 'l' : '.',
           status.isSet(G4PARTICLE_SIM_STOPPED) ? 's' : '.'
           );
#endif
}

/// Default destructor
Geant4ParticleMap::~Geant4ParticleMap()    {
  clear();
}

/// Clear particle maps
void Geant4ParticleMap::clear()    {
  detail::releaseObjects(particleMap);
  particleMap.clear();
  equivalentTracks.clear();
}

/// Dump content
void Geant4ParticleMap::dump()  const  {
  int cnt;
  char text[64];
  using namespace std;
  const Geant4ParticleMap* m = this;

  cnt = 0;
  cout << "Particle map:" << endl;
  for(Geant4ParticleMap::ParticleMap::const_iterator i=m->particleMap.begin(); i!=m->particleMap.end();++i)  {
    ::snprintf(text,sizeof(text)," [%-4d:%p]",(*i).second->id,(void*)(*i).second);
    cout << text;
    if ( ++cnt == 8 ) {
      cout << endl;
      cnt = 0;
    }
  }
  cout << endl;

  cnt = 0;
  cout << "Equivalents:" << endl;
  for(Geant4ParticleMap::TrackEquivalents::const_iterator i=m->equivalentTracks.begin(); i!=m->equivalentTracks.end();++i)  {
    ::snprintf(text,sizeof(text)," [%-5d : %-5d]",(*i).first,(*i).second);
    cout << text;
    if ( ++cnt == 8 ) {
      cout << endl;
      cnt = 0;
    }
  }
  cout << endl;
}

/// Adopt particle maps
void Geant4ParticleMap::adopt(ParticleMap& pm, TrackEquivalents& equiv)    {
  clear();
  particleMap = pm;
  equivalentTracks = equiv;
  pm.clear();
  equiv.clear();
  //dump();
}

/// Check if the particle map was ever filled (ie. some particle handler was present)
  bool Geant4ParticleMap::isValid() const   {
  return !equivalentTracks.empty();
}

/// Access the equivalent track id (shortcut to the usage of TrackEquivalents)
int Geant4ParticleMap::particleID(int g4_id, bool) const   {
  TrackEquivalents::const_iterator iequiv = equivalentTracks.find(g4_id);
  if ( iequiv != equivalentTracks.end() ) return (*iequiv).second;
  printout(ERROR,"Geant4ParticleMap","+++ No Equivalent particle for track:%d."
           " Monte Carlo truth record looks broken!",g4_id);
  dump();
  return -1;
}
