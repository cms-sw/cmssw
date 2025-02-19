#include "SimG4Core/CountProcesses/interface/CountProcessesAction.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/EndOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"

#include "G4Track.hh"
#include "G4Run.hh"
#include "G4Event.hh"
#include "G4Step.hh"
#include "G4ProcessManager.hh"
#include "G4ParticleTable.hh"

CountProcessesAction::CountProcessesAction(edm::ParameterSet const & p)
    : fDEBUG(p.getUntrackedParameter<bool>("DEBUG",false))
{}

CountProcessesAction::~CountProcessesAction() {}

void CountProcessesAction::update(const BeginOfRun * run)
{
    G4ParticleTable * partTable = G4ParticleTable::GetParticleTable();
    int siz = partTable->size();
    for (int ii= 0; ii < siz; ii++)
    {
	G4ParticleDefinition * particle = partTable->GetParticle(ii);
	std::string particleName = particle->GetParticleName();
	if (fDEBUG)
	    std::cout << ii << " PCA " << particleName<< " " << particle->GetPDGStable() 
		      << " " << particle->IsShortLived() << std::endl;
	theParticleList[particleName] = 0;

	//--- All processes of this particle 
	G4ProcessManager * pmanager = particle->GetProcessManager();
	G4ProcessVector * pvect = pmanager->GetProcessList();
	int sizproc = pvect->size();
	for (int jj = 0; jj < sizproc; jj++) 
	{
	    std::string processName = (*pvect)[jj]->GetProcessName();
	    if (fDEBUG)
		std::cout << jj << " PCR " << processName<< std::endl;
	    theProcessList[pss(particleName,processName)] = 0;
	}
    }
    DumpProcessList(0);
}

void CountProcessesAction::update(const BeginOfTrack * trk)
{
    //----- Fill counter of particles
    const G4Track * aTrack = (*trk)();
    std::string particleName = aTrack->GetDefinition()->GetParticleName();
    theParticleList[particleName]++;

    //----- Fill counter of Creator Processes
    const G4VProcess * proc = aTrack->GetCreatorProcess();
    std::string processName;
    if (proc != 0) processName = proc->GetProcessName();
    else processName = "Primary";
    pss parproc(particleName,processName);
    mpssi::iterator ite = theCreatorProcessList.find(parproc);
    if (ite == theCreatorProcessList.end()) theCreatorProcessList[ parproc ] = 1;
    else (*ite).second = (*ite).second +1; 
    if (fDEBUG) 
	std::cout << " creator " << particleName << " " << processName 
		  << theCreatorProcessList.size() << std::endl;
}

void CountProcessesAction::update(const G4Step* aStep )
{
    std::string processName;
    if(aStep->GetPostStepPoint()->GetProcessDefinedStep() != 0)
	processName = aStep->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();
    else processName = "User Limit";
    std::string particleName = aStep->GetTrack()->GetDefinition()->GetParticleName();
    theProcessList[pss(particleName,processName)] = theProcessList[pss(particleName,processName)] + 1;
}

void CountProcessesAction::update(const EndOfRun * run)
{
    DumpProcessList(1);
    DumpCreatorProcessList(1);
    DumpParticleList();
}

void CountProcessesAction::DumpProcessList(bool printNsteps, std::ostream& out)
{    
    mpssi::iterator ite;
    for (ite = theProcessList.begin(); ite != theProcessList.end(); ite++) 
    {
	if (!printNsteps) 
	    out << "PROC_LIST " << (*ite).first.first << " : " 
		<< (*ite) .first.second << std::endl; 
	else if ((*ite).second != 0)
	    out << "PROC_COUNT " << (*ite).first.first << " : " 
		<< (*ite) .first.second << " = " << (*ite).second << std::endl; 
    }
}

void CountProcessesAction::DumpCreatorProcessList(bool printNsteps, std::ostream& out)
{    
    mpssi::iterator ite;
    for (ite = theCreatorProcessList.begin(); ite != theCreatorProcessList.end(); ite++) 
    {
	if (!printNsteps) 
	    out << "PROC-CREATOR_LIST " << (*ite).first.first << " : " 
		<<(*ite) .first.second << std::endl; 
	else if ((*ite).second != 0) 
	    out << "PROC_CREATOR_COUNT " << (*ite).first.first << " : " 
		<<(*ite) .first.second << " = " << (*ite).second << std::endl; 
    }
}

void CountProcessesAction::DumpParticleList(std::ostream& out)
{    
    psi::iterator ite;
    for (ite = theParticleList.begin(); ite != theParticleList.end(); ite++) 
    {
	if ((*ite).second != 0) 
	    out << "PART_LIST: " << (*ite).first << " = " << (*ite).second << std::endl; 
    }
}

