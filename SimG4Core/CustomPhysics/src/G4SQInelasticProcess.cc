
#include "SimG4Core/CustomPhysics/interface/G4SQInelasticProcess.h"
#include "SimG4Core/CustomPhysics/interface/G4SQInelasticCrossSection.h"
#include "SimG4Core/CustomPhysics/interface/G4SQ.h"

#include "G4Types.hh"
#include "G4SystemOfUnits.hh"
#include "G4HadProjectile.hh"
#include "G4ElementVector.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4Element.hh"
#include "G4ParticleChange.hh"
#include "G4NucleiProperties.hh"
#include "G4Nucleus.hh"

#include "G4HadronicException.hh"
#include "G4HadronicProcessStore.hh"
#include "G4HadronicInteraction.hh"

#include "G4ParticleDefinition.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


G4SQInelasticProcess::G4SQInelasticProcess(double mass, const G4String& processName)
 : G4HadronicProcess(processName, fHadronic)
{
  AddDataSet(new G4SQInelasticCrossSection(mass));
  theParticle = G4SQ::SQ(mass);
}


G4SQInelasticProcess::~G4SQInelasticProcess()
{
}


G4bool G4SQInelasticProcess::IsApplicable(const G4ParticleDefinition& aP)
{
  return  theParticle->GetParticleType() == aP.GetParticleType();
}


G4VParticleChange*
G4SQInelasticProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
{

  edm::LogInfo("G4SQInelasticProcess::PostStepDoIt")
    << "Particle is going to interact at position" << aTrack.GetPosition()/cm 
    << " momentumdirection eta: " << aTrack.GetMomentumDirection().eta()
    << " interacting in material : " << aTrack.GetMaterial() << std::endl; 

  // if primary is not Alive then do nothing
  theTotalResult->Clear();
  theTotalResult->Initialize(aTrack);
  theTotalResult->ProposeWeight(aTrack.GetWeight());
  if(aTrack.GetTrackStatus() != fAlive) {
    edm::LogInfo("G4SQInelasticProcess::PostStepDoIt")
      << "No interaction because primary is not alive" << std::endl; 
    return theTotalResult;
  }


  edm::LogInfo("G4SQInelasticProcess::PostStepDoIt") 
    << "Start a possible interaction?" << std::endl;

  if(aTrack.GetPosition().rho()/centimeter < 1) {
    edm::LogInfo("G4SQInelasticProcess::PostStepDoIt")
      << "FYI: the rho of the track is < 1cm and it's still going to interact..." << std::endl;
  }

  // Find cross section at end of step and check if <= 0
  //
  G4DynamicParticle* aParticle = const_cast<G4DynamicParticle *>(aTrack.GetDynamicParticle());

  G4Material* aMaterial = aTrack.GetMaterial();

  const G4Element* anElement = 0;
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
    G4Exception("G4SQInelasticProcess::PostStepDoIt", "had003", FatalException,
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
      ed << "G4SQInelasticProcess: track in unusable state - "
	 << aTrack.GetTrackStatus() << G4endl;
      ed << "G4SQInelasticProcess: returning unchanged track " << G4endl;
      DumpState(aTrack,"PostStepDoIt",ed);
      G4Exception("G4SQInelasticProcess::PostStepDoIt", "had004", JustWarning, ed);
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
    theInteraction = GetHadronicInteractionList().at(0);
//    theInteraction = GetHadronicInteractionList()[0];
//      ChooseHadronicInteraction( kineticEnergy, aMaterial, anElement );
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
    G4Exception("G4SQInelasticProcess::PostStepDoIt", "had005", FatalException,
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
      // Call the interaction
      result = theInteraction->ApplyYourself( thePro, targetNucleus);
      ++reentryCount;
    }
    catch(G4HadronicException & aR)
    {
      G4ExceptionDescription ed;
      aR.Report(ed);
      ed << "Call for " << theInteraction->GetModelName() << G4endl;
      ed << "Target element "<<anElement->GetName()<<"  Z= "
	 << targetNucleus.GetZ_asInt()
	 << "  A= " << targetNucleus.GetA_asInt() << G4endl;
      DumpState(aTrack,"ApplyYourself",ed);
      ed << " ApplyYourself failed" << G4endl;
      G4Exception("G4SQInelasticProcess::PostStepDoIt", "had006", FatalException,
		  ed);
    }


    edm::LogInfo("G4SQInelasticProcess::PostStepDoIt")
      << "Call for " << theInteraction->GetModelName() 
      << std::endl;
    edm::LogInfo("G4SQInelasticProcess::PostStepDoIt")
      << "Target element "<< anElement->GetName()
      << "  Z=" << targetNucleus.GetZ_asInt()
      << "  A=" << targetNucleus.GetA_asInt() 
      << std::endl;

    edm::LogInfo("G4SQInelasticProcess::PostStepDoIt")
      << "Nr of secondaries: " << result->GetNumberOfSecondaries() << std::endl
      << "Momentum change and E deeposit: " << result->GetMomentumChange() << " " << result->GetLocalEnergyDeposit() << std::endl
      << "Track position and vertex: " << aTrack.GetPosition() << " " << aTrack.GetVertexPosition() << std::endl;

    float r = aTrack.GetPosition().perp();
    float z = fabs(aTrack.GetPosition().z());
    edm::LogInfo("G4SQInelasticProcess::PostStepDoIt")
      << "In tracker volume? "
      << (r<(100*cm) && z<(200*cm)? " YES!  " : " NO!  ")
      << "r=" << r/cm << " z=" << z/cm << std::endl;

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
      G4Exception("G4SQInelasticProcess::PostStepDoIt", "had006", FatalException,
		  ed);
    }
  }
  while(!result);

  edm::LogInfo("G4SQInelasticProcess::PostStepDoIt")
    << "=== Anti-sexaquark interaction succeeded!" << std::endl;
  result->SetTrafoToLab(thePro.GetTrafoToLab());

  ClearNumberOfInteractionLengthLeft();

  FillResult(result, aTrack);

  if (epReportLevel != 0) {
    CheckEnergyMomentumConservation(aTrack, targetNucleus);
  }
  return theTotalResult;
}


G4HadFinalState* G4SQInelasticProcess::CheckResult(const G4HadProjectile & aPro,const G4Nucleus &aNucleus, G4HadFinalState * result)
{
   // check for catastrophic energy non-conservation, to re-sample the interaction

   G4HadronicInteraction * theModel = GetHadronicInteractionList()[0];

   edm::LogInfo("G4SQInelasticProcess::PostStepDoIt")
     << "checkresult - " << theModel << std::endl;

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
         edm::LogInfo("G4SQInelasticProcess::PostStepDoIt")
           << "checkresult - interaction not complete: " << finalE << std::endl;
         if( nSec == 0 ){
            // Since there are no secondaries, there is no recoil nucleus.
            // To check energy balance we must neglect the initial nucleus too.
            nuclearMass=0.0;
         }
      }
      for (G4int i = 0; i < nSec; i++) {
         finalE += result->GetSecondary(i)->GetParticle()->GetTotalEnergy();
         edm::LogInfo("G4SQInelasticProcess::PostStepDoIt")
	   << "checkresult - secondary pdgId / E : " 
           << result->GetSecondary(i)->GetParticle()->GetPDGcode() << "\t"
           << result->GetSecondary(i)->GetParticle()->GetTotalEnergy() / GeV
           << std::endl;
      }
      G4double deltaE = nuclearMass + aPro.GetTotalEnergy() - finalE;
      edm::LogInfo("G4SQInelasticProcess::PostStepDoIt")
        << "checkresult - Total E: " << finalE / GeV << std::endl;
      edm::LogInfo("G4SQInelasticProcess::PostStepDoIt")
        << "checkresult - Energy balance: " << deltaE / GeV << std::endl;

      std::pair<G4double, G4double> checkLevels = theModel->GetFatalEnergyCheckLevels();	// (relative, absolute)
      if (std::abs(deltaE) > checkLevels.second && std::abs(deltaE) > checkLevels.first*aPro.GetKineticEnergy()){
         // do not delete result, this is a pointer to a data member;
         G4ExceptionDescription desc;
         desc << "Warning: Bad energy non-conservation detected, will "
              << (epReportLevel<0 ? "abort the event" :	"re-sample the interaction") << G4endl
              << " Process / Model: " <<  GetProcessName()<< " / " << theModel->GetModelName() << G4endl
              << " Primary: " << aPro.GetDefinition()->GetParticleName()
              << " (" << aPro.GetDefinition()->GetPDGEncoding() << "),"
              << " E= " <<  aPro.Get4Momentum().e()
              << ", target nucleus (" << aNucleus.GetZ_asInt() << ","<< aNucleus.GetA_asInt() << ")" << G4endl
              << " E(initial - final) = " << deltaE << " MeV." << G4endl;
         G4Exception("G4SQInelasticProcess:CheckResult()", "had012", epReportLevel<0 ? EventMustBeAborted : JustWarning,desc);
      }
   }
   return result;
}

