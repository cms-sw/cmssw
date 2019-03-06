
#include "G4SIMPInelasticProcess.hh"
#include "G4SIMP.hh"

#include "G4Types.hh"
#include "G4SystemOfUnits.hh"
#include "G4HadProjectile.hh"
#include "G4ElementVector.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4Element.hh"
#include "G4ParticleChange.hh"
#include "G4TransportationManager.hh"
#include "G4Navigator.hh"
#include "G4ProcessVector.hh"
#include "G4ProcessManager.hh"
#include "G4StableIsotopes.hh"
#include "G4HadTmpUtil.hh"
#include "G4NucleiProperties.hh"

#include "G4HadronicException.hh"
#include "G4HadronicProcessStore.hh"

#include <typeinfo>
#include <sstream>
#include <iostream>

#include <stdlib.h>

#include "G4HadronInelasticDataSet.hh"
#include "G4ParticleDefinition.hh"


// File-scope variable to capture environment variable at startup

static const char* G4Hadronic_Random_File = getenv("G4HADRONIC_RANDOM_FILE");

//////////////////////////////////////////////////////////////////
G4SIMPInelasticProcess::G4SIMPInelasticProcess(const G4String& processName)
 : G4HadronicProcess(processName, fHadronic)
{
  SetProcessSubType(fHadronInelastic);	// Default unless subclass changes
  
  theTotalResult = new G4ParticleChange();
  theTotalResult->SetSecondaryWeightByProcess(true);
  theInteraction = 0;
  theCrossSectionDataStore = new G4CrossSectionDataStore();
  G4HadronicProcessStore::Instance()->Register(this);
  aScaleFactor = 1;
  xBiasOn = false;
  G4SIMPInelasticProcess_debug_flag = false;

  GetEnergyMomentumCheckEnvvars();

  AddDataSet(new G4HadronInelasticDataSet());
  theParticle = G4SIMP::SIMP();

}


G4SIMPInelasticProcess::~G4SIMPInelasticProcess()
{
  G4HadronicProcessStore::Instance()->DeRegister(this);
  delete theTotalResult;
  delete theCrossSectionDataStore;
}


G4bool G4SIMPInelasticProcess::IsApplicable(const G4ParticleDefinition& aP)
{
  return  theParticle->GetParticleType() == aP.GetParticleType();
}


void G4SIMPInelasticProcess::GetEnergyMomentumCheckEnvvars() {
  levelsSetByProcess = false;

  epReportLevel = getenv("G4Hadronic_epReportLevel") ?
    strtol(getenv("G4Hadronic_epReportLevel"),0,10) : 0;

  epCheckLevels.first = getenv("G4Hadronic_epCheckRelativeLevel") ?
    strtod(getenv("G4Hadronic_epCheckRelativeLevel"),0) : DBL_MAX;

  epCheckLevels.second = getenv("G4Hadronic_epCheckAbsoluteLevel") ?
    strtod(getenv("G4Hadronic_epCheckAbsoluteLevel"),0) : DBL_MAX;
}

void G4SIMPInelasticProcess::RegisterMe( G4HadronicInteraction *a )
{
  if(!a) { return; }
  try{GetManagerPointer()->RegisterMe( a );}
  catch(G4HadronicException & aE)
  {
    G4ExceptionDescription ed;
    aE.Report(ed);
    ed << "Unrecoverable error in " << GetProcessName()
       << " to register " << a->GetModelName() << G4endl;
    G4Exception("G4SIMPInelasticProcess::RegisterMe", "had001", FatalException,
		ed);
  }
  G4HadronicProcessStore::Instance()->RegisterInteraction(this, a);
}

void G4SIMPInelasticProcess::PreparePhysicsTable(const G4ParticleDefinition& p)
{
  if(getenv("G4SIMPInelasticProcess_debug")) {
    G4SIMPInelasticProcess_debug_flag = true;
  }
  G4HadronicProcessStore::Instance()->RegisterParticle(this, &p);
}

void G4SIMPInelasticProcess::BuildPhysicsTable(const G4ParticleDefinition& p)
{
  try
  {
    theCrossSectionDataStore->BuildPhysicsTable(p);
  }
  catch(G4HadronicException aR)
  {
    G4ExceptionDescription ed;
    aR.Report(ed);
    ed << " hadronic initialisation fails" << G4endl;
    G4Exception("G4SIMPInelasticProcess::BuildPhysicsTable", "had000", 
		FatalException,ed);
  }
  G4HadronicProcessStore::Instance()->PrintInfo(&p);
}

G4double G4SIMPInelasticProcess::
GetMeanFreePath(const G4Track &aTrack, G4double, G4ForceCondition *)
{
  try
  {
    theLastCrossSection = aScaleFactor*
      theCrossSectionDataStore->GetCrossSection(aTrack.GetDynamicParticle(),
						aTrack.GetMaterial());
  }
  catch(G4HadronicException aR)
  {
    G4ExceptionDescription ed;
    aR.Report(ed);
    DumpState(aTrack,"GetMeanFreePath",ed);
    ed << " Cross section is not available" << G4endl;
    G4Exception("G4SIMPInelasticProcess::GetMeanFreePath", "had002", FatalException,
		ed);
  }
  G4double res = DBL_MAX;
  if( theLastCrossSection > 0.0 ) { res = 1.0/theLastCrossSection; }
  return res;
}

G4VParticleChange*
G4SIMPInelasticProcess::PostStepDoIt(const G4Track& aTrack, const G4Step&)
{

  // if primary is not Alive then do nothing
  theTotalResult->Clear();
  theTotalResult->Initialize(aTrack);
  theTotalResult->ProposeWeight(aTrack.GetWeight());
  if(aTrack.GetTrackStatus() != fAlive) { return theTotalResult; }

  // Find cross section at end of step and check if <= 0
  //
  G4DynamicParticle* aParticle = const_cast<G4DynamicParticle *>(aTrack.GetDynamicParticle());

  // change this SIMP particle in a neutron
  aParticle->SetPDGcode(2112);
  aParticle->SetDefinition(G4Neutron::Neutron());

  G4Material* aMaterial = aTrack.GetMaterial();

  G4Element* anElement = 0;
  try
  {
     anElement = theCrossSectionDataStore->SampleZandA(aParticle,
						       aMaterial,
						       targetNucleus);
  }
  catch(G4HadronicException & aR)
  {
    G4ExceptionDescription ed;
    aR.Report(ed);
    DumpState(aTrack,"SampleZandA",ed);
    ed << " PostStepDoIt failed on element selection" << G4endl;
    G4Exception("G4SIMPInelasticProcess::PostStepDoIt", "had003", FatalException,
		ed);
  }

  // check only for charged particles
  if(aParticle->GetDefinition()->GetPDGCharge() != 0.0) {
    if (GetElementCrossSection(aParticle, anElement, aMaterial) <= 0.0) {
      // No interaction
      return theTotalResult;
    }    
  }

  // Next check for illegal track status
  //
  if (aTrack.GetTrackStatus() != fAlive && aTrack.GetTrackStatus() != fSuspend) {
    if (aTrack.GetTrackStatus() == fStopAndKill ||
        aTrack.GetTrackStatus() == fKillTrackAndSecondaries ||
        aTrack.GetTrackStatus() == fPostponeToNextEvent) {
      G4ExceptionDescription ed;
      ed << "G4SIMPInelasticProcess: track in unusable state - "
	 << aTrack.GetTrackStatus() << G4endl;
      ed << "G4SIMPInelasticProcess: returning unchanged track " << G4endl;
      DumpState(aTrack,"PostStepDoIt",ed);
      G4Exception("G4SIMPInelasticProcess::PostStepDoIt", "had004", JustWarning, ed);
    }
    // No warning for fStopButAlive which is a legal status here
    return theTotalResult;
  }

  // Go on to regular case
  //
  G4double originalEnergy = aParticle->GetKineticEnergy();
  G4double kineticEnergy = originalEnergy;

  // Get kinetic energy per nucleon for ions
  if(aParticle->GetParticleDefinition()->GetBaryonNumber() > 1.5)
          kineticEnergy/=aParticle->GetParticleDefinition()->GetBaryonNumber();

  try
  {
    theInteraction =
      ChooseHadronicInteraction( kineticEnergy, aMaterial, anElement );
  }
  catch(G4HadronicException & aE)
  {
    G4ExceptionDescription ed;
    aE.Report(ed);
    ed << "Target element "<<anElement->GetName()<<"  Z= "
       << targetNucleus.GetZ_asInt() << "  A= "
       << targetNucleus.GetA_asInt() << G4endl;
    DumpState(aTrack,"ChooseHadronicInteraction",ed);
    ed << " No HadronicInteraction found out" << G4endl;
    G4Exception("G4SIMPInelasticProcess::PostStepDoIt", "had005", FatalException,
		ed);
  }

  // Initialize the hadronic projectile from the track
  thePro.Initialise(aTrack);
  G4HadFinalState* result = 0;
  G4int reentryCount = 0;

  do
  {
    try
    {
      // Save random engine if requested for debugging
      if (G4Hadronic_Random_File) {
         CLHEP::HepRandom::saveEngineStatus(G4Hadronic_Random_File);
      }
      // Call the interaction
      result = theInteraction->ApplyYourself( thePro, targetNucleus);
      ++reentryCount;
    }
    catch(G4HadronicException aR)
    {
      G4ExceptionDescription ed;
      aR.Report(ed);
      ed << "Call for " << theInteraction->GetModelName() << G4endl;
      ed << "Target element "<<anElement->GetName()<<"  Z= "
	 << targetNucleus.GetZ_asInt()
	 << "  A= " << targetNucleus.GetA_asInt() << G4endl;
      DumpState(aTrack,"ApplyYourself",ed);
      ed << " ApplyYourself failed" << G4endl;
      G4Exception("G4SIMPInelasticProcess::PostStepDoIt", "had006", FatalException,
		  ed);
    }

    // Check the result for catastrophic energy non-conservation
    result = CheckResult(thePro,targetNucleus, result);
    if(reentryCount>100) {
      G4ExceptionDescription ed;
      ed << "Call for " << theInteraction->GetModelName() << G4endl;
      ed << "Target element "<<anElement->GetName()<<"  Z= "
	 << targetNucleus.GetZ_asInt()
	 << "  A= " << targetNucleus.GetA_asInt() << G4endl;
      DumpState(aTrack,"ApplyYourself",ed);
      ed << " ApplyYourself does not completed after 100 attempts" << G4endl;
      G4Exception("G4SIMPInelasticProcess::PostStepDoIt", "had006", FatalException,
		  ed);
    }
  }
  while(!result);

  result->SetTrafoToLab(thePro.GetTrafoToLab());

  ClearNumberOfInteractionLengthLeft();

  FillResult(result, aTrack);

  if (epReportLevel != 0) {
    CheckEnergyMomentumConservation(aTrack, targetNucleus);
  }
  return theTotalResult;
}


void G4SIMPInelasticProcess::ProcessDescription(std::ostream& outFile) const
{
  outFile << "The description for this process has not been written yet.\n";
}


G4double G4SIMPInelasticProcess::XBiasSurvivalProbability()
{
  G4double result = 0;
  G4double nLTraversed = GetTotalNumberOfInteractionLengthTraversed();
  G4double biasedProbability = 1.-std::exp(-nLTraversed);
  G4double realProbability = 1-std::exp(-nLTraversed/aScaleFactor);
  result = (biasedProbability-realProbability)/biasedProbability;
  return result;
}

G4double G4SIMPInelasticProcess::XBiasSecondaryWeight()
{
  G4double result = 0;
  G4double nLTraversed = GetTotalNumberOfInteractionLengthTraversed();
  result =
     1./aScaleFactor*std::exp(-nLTraversed/aScaleFactor*(1-1./aScaleFactor));
  return result;
}

void
G4SIMPInelasticProcess::FillResult(G4HadFinalState * aR, const G4Track & aT)
{
  theTotalResult->ProposeLocalEnergyDeposit(aR->GetLocalEnergyDeposit());

  G4double rotation = CLHEP::twopi*G4UniformRand();
  G4ThreeVector it(0., 0., 1.);

  G4double efinal = aR->GetEnergyChange();
  if(efinal < 0.0) { efinal = 0.0; }

  // check status of primary
  if(aR->GetStatusChange() == stopAndKill) {
    theTotalResult->ProposeTrackStatus(fStopAndKill);
    theTotalResult->ProposeEnergy( 0.0 );

    // check its final energy
  } else if(0.0 == efinal) {
    theTotalResult->ProposeEnergy( 0.0 );
    if(aT.GetParticleDefinition()->GetProcessManager()
       ->GetAtRestProcessVector()->size() > 0)
         { aParticleChange.ProposeTrackStatus(fStopButAlive); }
    else { aParticleChange.ProposeTrackStatus(fStopAndKill); }

    // primary is not killed apply rotation and Lorentz transformation
  } else  {
    theTotalResult->ProposeTrackStatus(fAlive);
    G4double mass = aT.GetParticleDefinition()->GetPDGMass();
    G4double newE = efinal + mass;
    G4double newP = std::sqrt(efinal*(efinal + 2*mass));
    G4ThreeVector newPV = newP*aR->GetMomentumChange();
    G4LorentzVector newP4(newE, newPV);
    newP4.rotate(rotation, it);
    newP4 *= aR->GetTrafoToLab();
    theTotalResult->ProposeMomentumDirection(newP4.vect().unit());
    newE = newP4.e() - mass;
    if(G4SIMPInelasticProcess_debug_flag && newE <= 0.0) {
      G4ExceptionDescription ed;
      DumpState(aT,"Primary has zero energy after interaction",ed);
      G4Exception("G4SIMPInelasticProcess::FillResults", "had011", JustWarning, ed);
    }
    if(newE < 0.0) { newE = 0.0; }
    theTotalResult->ProposeEnergy( newE );
  }

  // check secondaries: apply rotation and Lorentz transformation
  G4int nSec = aR->GetNumberOfSecondaries();
  theTotalResult->SetNumberOfSecondaries(nSec);
  G4double weight = aT.GetWeight();

  if (nSec > 0) {
    G4double time0 = aT.GetGlobalTime();
    for (G4int i = 0; i < nSec; ++i) {
      G4LorentzVector theM = aR->GetSecondary(i)->GetParticle()->Get4Momentum();
      theM.rotate(rotation, it);
      theM *= aR->GetTrafoToLab();
      aR->GetSecondary(i)->GetParticle()->Set4Momentum(theM);

      // time of interaction starts from zero
      G4double time = aR->GetSecondary(i)->GetTime();
      if (time < 0.0) { time = 0.0; }

      // take into account global time
      time += time0;

      G4Track* track = new G4Track(aR->GetSecondary(i)->GetParticle(),
                                   time, aT.GetPosition());
      G4double newWeight = weight*aR->GetSecondary(i)->GetWeight();
	// G4cout << "#### ParticleDebug "
	// <<GetProcessName()<<" "
	// <<aR->GetSecondary(i)->GetParticle()->GetDefinition()->GetParticleName()<<" "
	// <<aScaleFactor<<" "
	// <<XBiasSurvivalProbability()<<" "
	// <<XBiasSecondaryWeight()<<" "
	// <<aT.GetWeight()<<" "
	// <<aR->GetSecondary(i)->GetWeight()<<" "
	// <<aR->GetSecondary(i)->GetParticle()->Get4Momentum()<<" "
	// <<G4endl;
      track->SetWeight(newWeight);
      track->SetTouchableHandle(aT.GetTouchableHandle());
      theTotalResult->AddSecondary(track);
      if (G4SIMPInelasticProcess_debug_flag) {
        G4double e = track->GetKineticEnergy();
        if (e <= 0.0) {
          G4ExceptionDescription ed;
          DumpState(aT,"Secondary has zero energy",ed);
          ed << "Secondary " << track->GetDefinition()->GetParticleName()
             << G4endl;
          G4Exception("G4SIMPInelasticProcess::FillResults", "had011", JustWarning,ed);
        }
      }
    }
  }

  aR->Clear();
  return;
}
/*
void
G4SIMPInelasticProcess::FillTotalResult(G4HadFinalState* aR, const G4Track& aT)
{
  theTotalResult->Clear();
  theTotalResult->ProposeLocalEnergyDeposit(0.);
  theTotalResult->Initialize(aT);
  theTotalResult->SetSecondaryWeightByProcess(true);
  theTotalResult->ProposeTrackStatus(fAlive);
  G4double rotation = CLHEP::twopi*G4UniformRand();
  G4ThreeVector it(0., 0., 1.);

  if(aR->GetStatusChange()==stopAndKill)
  {
    if( xBiasOn && G4UniformRand()<XBiasSurvivalProbability() )
    {
      theTotalResult->ProposeParentWeight( XBiasSurvivalProbability()*aT.GetWeight() );
    }
    else
    {
      theTotalResult->ProposeTrackStatus(fStopAndKill);
      theTotalResult->ProposeEnergy( 0.0 );
    }
  }
  else if(aR->GetStatusChange()!=stopAndKill )
  {
    if(aR->GetStatusChange()==suspend)
    {
      theTotalResult->ProposeTrackStatus(fSuspend);
      if(xBiasOn)
      {
	G4ExceptionDescription ed;
        DumpState(aT,"FillTotalResult",ed);
        G4Exception("G4SIMPInelasticProcess::FillTotalResult", "had007", FatalException,
		    ed,"Cannot cross-section bias a process that suspends tracks.");
      }
    } else if (aT.GetKineticEnergy() == 0) {
      theTotalResult->ProposeTrackStatus(fStopButAlive);
    }

    if(xBiasOn && G4UniformRand()<XBiasSurvivalProbability())
    {
      theTotalResult->ProposeParentWeight( XBiasSurvivalProbability()*aT.GetWeight() );
      G4double newWeight = aR->GetWeightChange()*aT.GetWeight();
      G4double newM=aT.GetParticleDefinition()->GetPDGMass();
      G4double newE=aR->GetEnergyChange() + newM;
      G4double newP=std::sqrt(newE*newE - newM*newM);
      G4DynamicParticle * aNew =
      new G4DynamicParticle(aT.GetParticleDefinition(), newE, newP*aR->GetMomentumChange());
      aR->AddSecondary(G4HadSecondary(aNew, newWeight));
    }
    else
    {
      G4double newWeight = aR->GetWeightChange()*aT.GetWeight();
      theTotalResult->ProposeParentWeight(newWeight); // This is multiplicative
      if(aR->GetEnergyChange()>-.5)
      {
        theTotalResult->ProposeEnergy(aR->GetEnergyChange());
      }
      G4LorentzVector newDirection(aR->GetMomentumChange().unit(), 1.);
      newDirection*=aR->GetTrafoToLab();
      theTotalResult->ProposeMomentumDirection(newDirection.vect());
    }
  }
  else
  {
    G4ExceptionDescription ed;
    ed << "Call for " << theInteraction->GetModelName() << G4endl;
    ed << "Target Z= " 
	   << targetNucleus.GetZ_asInt() 
	   << "  A= " << targetNucleus.GetA_asInt() << G4endl;
    DumpState(aT,"FillTotalResult",ed);
    G4Exception("G4SIMPInelasticProcess", "had008", FatalException,
    "use of unsupported track-status.");
  }

  if(GetProcessName() != "hElastic" && GetProcessName() != "HadronElastic"
     &&  theTotalResult->GetTrackStatus()==fAlive
     && aR->GetStatusChange()==isAlive)
    {
    // Use for debugging:   G4double newWeight = theTotalResult->GetParentWeight();

    G4double newKE = std::max(DBL_MIN, aR->GetEnergyChange());
    G4DynamicParticle* aNew = new G4DynamicParticle(aT.GetParticleDefinition(),
                                                    aR->GetMomentumChange(),
                                                    newKE);
    aR->AddSecondary(aNew);
    aR->SetStatusChange(stopAndKill);

    theTotalResult->ProposeTrackStatus(fStopAndKill);
    theTotalResult->ProposeEnergy( 0.0 );

  }
  theTotalResult->ProposeLocalEnergyDeposit(aR->GetLocalEnergyDeposit());
  theTotalResult->SetNumberOfSecondaries(aR->GetNumberOfSecondaries());

  if(aR->GetStatusChange() != stopAndKill)
  {
    G4double newM=aT.GetParticleDefinition()->GetPDGMass();
    G4double newE=aR->GetEnergyChange() + newM;
    G4double newP=std::sqrt(newE*newE - newM*newM);
    G4ThreeVector newPV = newP*aR->GetMomentumChange();
    G4LorentzVector newP4(newE, newPV);
    newP4.rotate(rotation, it);
    newP4*=aR->GetTrafoToLab();
    theTotalResult->ProposeMomentumDirection(newP4.vect().unit());
  }

  for(G4int i=0; i<aR->GetNumberOfSecondaries(); ++i)
  {
    G4LorentzVector theM = aR->GetSecondary(i)->GetParticle()->Get4Momentum();
    theM.rotate(rotation, it);
    theM*=aR->GetTrafoToLab();
    aR->GetSecondary(i)->GetParticle()->Set4Momentum(theM);
    G4double time = aR->GetSecondary(i)->GetTime();
    if(time<0) time = aT.GetGlobalTime();

    G4Track* track = new G4Track(aR->GetSecondary(i)->GetParticle(),
				 time,
				 aT.GetPosition());

    G4double newWeight = aT.GetWeight()*aR->GetSecondary(i)->GetWeight();
    if(xBiasOn) { newWeight *= XBiasSecondaryWeight(); }
    track->SetWeight(newWeight);
    track->SetTouchableHandle(aT.GetTouchableHandle());
    theTotalResult->AddSecondary(track);
  }

  aR->Clear();
  return;
}
*/

void G4SIMPInelasticProcess::BiasCrossSectionByFactor(G4double aScale)
{
  xBiasOn = true;
  aScaleFactor = aScale;
  G4String it = GetProcessName();
  if( (it != "PhotonInelastic") &&
      (it != "ElectroNuclear") &&
      (it != "PositronNuclear") )
    {
      G4ExceptionDescription ed;
      G4Exception("G4SIMPInelasticProcess::BiasCrossSectionByFactor", "had009", FatalException, ed,
		  "Cross-section biasing available only for gamma and electro nuclear reactions.");
    }
  if(aScale<100)
    {
      G4ExceptionDescription ed;
      G4Exception("G4SIMPInelasticProcess::BiasCrossSectionByFactor", "had010", JustWarning,ed,
		  "Cross-section bias readjusted to be above safe limit. New value is 100");
      aScaleFactor = 100.;
    }
}

G4HadFinalState* G4SIMPInelasticProcess::CheckResult(const G4HadProjectile & aPro,const G4Nucleus &aNucleus, G4HadFinalState * result) const
{
   // check for catastrophic energy non-conservation, to re-sample the interaction

   G4HadronicInteraction * theModel = GetHadronicInteraction();
   G4double nuclearMass(0);
   if (theModel){

      // Compute final-state total energy
      G4double finalE(0.);
      G4int nSec = result->GetNumberOfSecondaries();

      nuclearMass = G4NucleiProperties::GetNuclearMass(aNucleus.GetA_asInt(),
                                                       aNucleus.GetZ_asInt());
      if (result->GetStatusChange() != stopAndKill) {
       	// Interaction didn't complete, returned "do nothing" state          => reset nucleus
        //  or  the primary survived the interaction (e.g. electro-nuclear ) => keep  nucleus
         finalE=result->GetLocalEnergyDeposit() +
		aPro.GetDefinition()->GetPDGMass() + result->GetEnergyChange();
         if( nSec == 0 ){
            // Since there are no secondaries, there is no recoil nucleus.
            // To check energy balance we must neglect the initial nucleus too.
            nuclearMass=0.0;
         }
      }
      for (G4int i = 0; i < nSec; i++) {
         finalE += result->GetSecondary(i)->GetParticle()->GetTotalEnergy();
      }
      G4double deltaE= nuclearMass +  aPro.GetTotalEnergy() -  finalE;

      std::pair<G4double, G4double> checkLevels = theModel->GetFatalEnergyCheckLevels();	// (relative, absolute)
      if (std::abs(deltaE) > checkLevels.second && std::abs(deltaE) > checkLevels.first*aPro.GetKineticEnergy()){
         // do not delete result, this is a pointer to a data member;
         result=0;
         G4ExceptionDescription desc;
         desc << "Warning: Bad energy non-conservation detected, will "
              << (epReportLevel<0 ? "abort the event" :	"re-sample the interaction") << G4endl
              << " Process / Model: " <<  GetProcessName()<< " / " << theModel->GetModelName() << G4endl
              << " Primary: " << aPro.GetDefinition()->GetParticleName()
              << " (" << aPro.GetDefinition()->GetPDGEncoding() << "),"
              << " E= " <<  aPro.Get4Momentum().e()
              << ", target nucleus (" << aNucleus.GetZ_asInt() << ","<< aNucleus.GetA_asInt() << ")" << G4endl
              << " E(initial - final) = " << deltaE << " MeV." << G4endl;
         G4Exception("G4SIMPInelasticProcess:CheckResult()", "had012", epReportLevel<0 ? EventMustBeAborted : JustWarning,desc);
      }
   }
   return result;
}

void
G4SIMPInelasticProcess::CheckEnergyMomentumConservation(const G4Track& aTrack,
                                                   const G4Nucleus& aNucleus)
{
  G4int target_A=aNucleus.GetA_asInt();
  G4int target_Z=aNucleus.GetZ_asInt();
  G4double targetMass = G4NucleiProperties::GetNuclearMass(target_A,target_Z);
  G4LorentzVector target4mom(0, 0, 0, targetMass);

  G4LorentzVector projectile4mom = aTrack.GetDynamicParticle()->Get4Momentum();
  G4int track_A = aTrack.GetDefinition()->GetBaryonNumber();
  G4int track_Z = G4lrint(aTrack.GetDefinition()->GetPDGCharge());

  G4int initial_A = target_A + track_A;
  G4int initial_Z = target_Z + track_Z;

  G4LorentzVector initial4mom = projectile4mom + target4mom;

  // Compute final-state momentum for scattering and "do nothing" results
  G4LorentzVector final4mom;
  G4int final_A(0), final_Z(0);

  G4int nSec = theTotalResult->GetNumberOfSecondaries();
  if (theTotalResult->GetTrackStatus() != fStopAndKill) {  // If it is Alive
     // Either interaction didn't complete, returned "do nothing" state
     //  or    the primary survived the interaction (e.g. electro-nucleus )
     G4Track temp(aTrack);

     // Use the final energy / momentum
     temp.SetMomentumDirection(*theTotalResult->GetMomentumDirection());
     temp.SetKineticEnergy(theTotalResult->GetEnergy());

     if( nSec == 0 ){
        // Interaction didn't complete, returned "do nothing" state
        //   - or suppressed recoil  (e.g. Neutron elastic )
        final4mom = temp.GetDynamicParticle()->Get4Momentum() + target4mom;
        final_A = initial_A;
        final_Z = initial_Z;
     }else{
        // The primary remains in final state (e.g. electro-nucleus )
        final4mom = temp.GetDynamicParticle()->Get4Momentum();
        final_A = track_A;
        final_Z = track_Z;
        // Expect that the target nucleus will have interacted,
        //  and its products, including recoil, will be included in secondaries.
     }
  }
  if( nSec > 0 ) {
    G4Track* sec;

    for (G4int i = 0; i < nSec; i++) {
      sec = theTotalResult->GetSecondary(i);
      final4mom += sec->GetDynamicParticle()->Get4Momentum();
      final_A += sec->GetDefinition()->GetBaryonNumber();
      final_Z += G4lrint(sec->GetDefinition()->GetPDGCharge());
    }
  }

  // Get level-checking information (used to cut-off relative checks)
  G4String processName = GetProcessName();
  G4HadronicInteraction* theModel = GetHadronicInteraction();
  G4String modelName("none");
  if (theModel) modelName = theModel->GetModelName();
  std::pair<G4double, G4double> checkLevels = epCheckLevels;
  if (!levelsSetByProcess) {
    if (theModel) checkLevels = theModel->GetEnergyMomentumCheckLevels();
    checkLevels.first= std::min(checkLevels.first,  epCheckLevels.first);
    checkLevels.second=std::min(checkLevels.second, epCheckLevels.second);
  }

  // Compute absolute total-energy difference, and relative kinetic-energy
  G4bool checkRelative = (aTrack.GetKineticEnergy() > checkLevels.second);

  G4LorentzVector diff = initial4mom - final4mom;
  G4double absolute = diff.e();
  G4double relative = checkRelative ? absolute/aTrack.GetKineticEnergy() : 0.;

  G4double absolute_mom = diff.vect().mag();
  G4double relative_mom = checkRelative ? absolute_mom/aTrack.GetMomentum().mag() : 0.;

  // Evaluate relative and absolute conservation
  G4bool relPass = true;
  G4String relResult = "pass";
  if (  std::abs(relative) > checkLevels.first
	 || std::abs(relative_mom) > checkLevels.first) {
    relPass = false;
    relResult = checkRelative ? "fail" : "N/A";
  }

  G4bool absPass = true;
  G4String absResult = "pass";
  if (   std::abs(absolute) > checkLevels.second
      || std::abs(absolute_mom) > checkLevels.second ) {
    absPass = false ;
    absResult = "fail";
  }

  G4bool chargePass = true;
  G4String chargeResult = "pass";
  if (   (initial_A-final_A)!=0
      || (initial_Z-final_Z)!=0 ) {
    chargePass = checkLevels.second < DBL_MAX ? false : true;
    chargeResult = "fail";
   }

  G4bool conservationPass = (relPass || absPass) && chargePass;

  std::stringstream Myout;
  G4bool Myout_notempty(false);
  // Options for level of reporting detail:
  //  0. off
  //  1. report only when E/p not conserved
  //  2. report regardless of E/p conservation
  //  3. report only when E/p not conserved, with model names, process names, and limits
  //  4. report regardless of E/p conservation, with model names, process names, and limits
  //  negative -1.., as above, but send output to stderr

  if(   std::abs(epReportLevel) == 4
	||	( std::abs(epReportLevel) == 3 && ! conservationPass ) ){
      Myout << " Process: " << processName << " , Model: " <<  modelName << G4endl;
      Myout << " Primary: " << aTrack.GetParticleDefinition()->GetParticleName()
            << " (" << aTrack.GetParticleDefinition()->GetPDGEncoding() << "),"
            << " E= " <<  aTrack.GetDynamicParticle()->Get4Momentum().e()
	    << ", target nucleus (" << aNucleus.GetZ_asInt() << ","
	    << aNucleus.GetA_asInt() << ")" << G4endl;
      Myout_notempty=true;
  }
  if (  std::abs(epReportLevel) == 4
	 || std::abs(epReportLevel) == 2
	 || ! conservationPass ){

      Myout << "   "<< relResult  <<" relative, limit " << checkLevels.first << ", values E/T(0) = "
             << relative << " p/p(0)= " << relative_mom  << G4endl;
      Myout << "   "<< absResult << " absolute, limit (MeV) " << checkLevels.second/MeV << ", values E / p (MeV) = "
             << absolute/MeV << " / " << absolute_mom/MeV << " 3mom: " << (diff.vect())*1./MeV <<  G4endl;
      Myout << "   "<< chargeResult << " charge/baryon number balance " << (initial_Z-final_Z) << " / " << (initial_A-final_A) << " "<<  G4endl;
      Myout_notempty=true;

  }
  Myout.flush();
  if ( Myout_notempty ) {
     if (epReportLevel > 0)      G4cout << Myout.str()<< G4endl;
     else if (epReportLevel < 0) G4cerr << Myout.str()<< G4endl;
  }
}


void G4SIMPInelasticProcess::DumpState(const G4Track& aTrack,
				  const G4String& method,
				  G4ExceptionDescription& ed)
{
  ed << "Unrecoverable error in the method " << method << " of "
     << GetProcessName() << G4endl;
  ed << "TrackID= "<< aTrack.GetTrackID() << "  ParentID= "
     << aTrack.GetParentID()
     << "  " << aTrack.GetParticleDefinition()->GetParticleName()
     << G4endl;
  ed << "Ekin(GeV)= " << aTrack.GetKineticEnergy()/CLHEP::GeV
     << ";  direction= " << aTrack.GetMomentumDirection() << G4endl;
  ed << "Position(mm)= " << aTrack.GetPosition()/CLHEP::mm << ";";

  if (aTrack.GetMaterial()) {
    ed << "  material " << aTrack.GetMaterial()->GetName();
  }
  ed << G4endl;

  if (aTrack.GetVolume()) {
    ed << "PhysicalVolume  <" << aTrack.GetVolume()->GetName()
       << ">" << G4endl;
  }
}
/* 
G4ParticleDefinition* G4SIMPInelasticProcess::GetTargetDefinition()
{
  const G4Nucleus* nuc = GetTargetNucleus();
  G4int Z = nuc->GetZ_asInt();
  G4int A = nuc->GetA_asInt();
  return G4ParticleTable::GetParticleTable()->GetIon(Z,A,0*eV);
}
*/
/* end of file */
