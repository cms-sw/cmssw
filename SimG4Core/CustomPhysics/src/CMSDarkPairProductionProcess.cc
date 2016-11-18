//
// ********************************************************************
// Authors of this file: Dustin Stolp (dostolp@ucdavis.edu)
//                       Sushil S. Chauhan (schauhan@cern.ch)   
//
// -----------------------------------------------------------------------------

#include "SimG4Core/CustomPhysics/interface/CMSDarkPairProductionProcess.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4BetheHeitlerModel.hh"
#include "G4PairProductionRelModel.hh"
#include "G4Electron.hh"


using namespace std;

static G4double darkFactor;

CMSDarkPairProductionProcess::CMSDarkPairProductionProcess(
  G4double df,
  const G4String& processName,  
  G4ProcessType type):G4VEmProcess (processName, type),
    isInitialised(false)
{ 
  darkFactor = df;
  SetMinKinEnergy(2.0*electron_mass_c2);
  SetProcessSubType(fGammaConversion);
  SetStartFromNullFlag(true);
  SetBuildTableFlag(true);
  SetSecondaryParticle(G4Electron::Electron());
  SetLambdaBinning(220);
}

 
CMSDarkPairProductionProcess::~CMSDarkPairProductionProcess()
{}


G4bool CMSDarkPairProductionProcess::IsApplicable(const G4ParticleDefinition& p)
{
  return (p.GetParticleType()=="darkpho");
}


void CMSDarkPairProductionProcess::InitialiseProcess(const G4ParticleDefinition* p)
{
  if(!isInitialised) {
    isInitialised = true;
    
       AddEmModel(0, new CMSDarkPairProduction(p,darkFactor));

  }
}


G4double CMSDarkPairProductionProcess::MinPrimaryEnergy(const G4ParticleDefinition*,
					     const G4Material*)
{
  return 2*electron_mass_c2;
}


void CMSDarkPairProductionProcess::PrintInfo()
{}         

