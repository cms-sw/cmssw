#ifndef SimG4Core_CountProcessesAction_H
#define SimG4Core_CountProcessesAction_H

#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
   
#include "G4ParticleDefinition.hh"

#include <string>
#include <map>

typedef std::map 
<std::pair<std::string,std::string>, int, std::less< std::pair<std::string,std::string> > > mpssi;
typedef std::pair<std::string,std::string> pss;
typedef std::map<std::string,int,std::less<std::string> > psi;

class BeginOfRun;
class EndOfRun;
class BeginOfTrack;
class G4Step;

class CountProcessesAction : public SimWatcher,
			     public Observer<const BeginOfRun *>, 
			     public Observer<const EndOfRun *>,  
			     public Observer<const BeginOfTrack *>, 
			     public Observer<const G4Step *> 
{
public:
    CountProcessesAction(edm::ParameterSet const & p);
    ~CountProcessesAction();
    void update(const BeginOfRun * run);
    void update(const BeginOfTrack * trk);
    void update(const EndOfRun * track);
    void update(const G4Step * track);
    //---- Dump list of processes for each particle. 
    // printNsteps = 1 print in how many step the process was called, 
    // print only those processes with this number <> 0
    void DumpProcessList(bool printNsteps, std::ostream& out = std::cout);
    void DumpCreatorProcessList(bool printNsteps, std::ostream& out = std::cout);
    void DumpParticleList(std::ostream& out = std::cout);
private:
    bool fDEBUG;
    mpssi theProcessList;
    mpssi theCreatorProcessList;
    psi theParticleList;
};

#endif
