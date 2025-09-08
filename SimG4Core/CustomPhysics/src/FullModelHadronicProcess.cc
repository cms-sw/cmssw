#include "G4ProcessManager.hh"
#include "G4ParticleTable.hh"
#include "G4HadronicException.hh"

#include "SimG4Core/CustomPhysics/interface/FullModelHadronicProcess.h"
#include "SimG4Core/CustomPhysics/interface/G4ProcessHelper.h"
#include "SimG4Core/CustomPhysics/interface/Decay3Body.h"
#include "SimG4Core/CustomPhysics/interface/CustomPDGParser.h"
#include "SimG4Core/CustomPhysics/interface/CustomParticle.h"

using namespace CLHEP;

FullModelHadronicProcess::FullModelHadronicProcess(G4ProcessHelper* aHelper, const G4String& processName)
    : G4VDiscreteProcess(processName), theHelper(aHelper) {}

FullModelHadronicProcess::~FullModelHadronicProcess() {}

G4bool FullModelHadronicProcess::IsApplicable(const G4ParticleDefinition& aP) {
  return theHelper->ApplicabilityTester(aP);
}

G4double FullModelHadronicProcess::GetMicroscopicCrossSection(const G4DynamicParticle* aParticle,
                                                              const G4Element* anElement,
                                                              G4double aTemp) {
  //Get the cross section for this particle/element combination from the ProcessHelper
  G4double InclXsec = theHelper->GetInclusiveCrossSection(aParticle, anElement);
  return InclXsec;
}

G4double FullModelHadronicProcess::GetMeanFreePath(const G4Track& aTrack, G4double, G4ForceCondition*) {
  G4Material* aMaterial = aTrack.GetMaterial();
  const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
  G4double sigma = 0.0;

  G4int nElements = aMaterial->GetNumberOfElements();

  const G4double* theAtomicNumDensityVector = aMaterial->GetAtomicNumDensityVector();
  G4double aTemp = aMaterial->GetTemperature();

  for (G4int i = 0; i < nElements; ++i) {
    G4double xSection = GetMicroscopicCrossSection(aParticle, (*aMaterial->GetElementVector())[i], aTemp);
    sigma += theAtomicNumDensityVector[i] * xSection;
  }
  G4double res = DBL_MAX;
  if (sigma > 0.0) {
    res = 1. / sigma;
  }
  return res;
}

G4VParticleChange* FullModelHadronicProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep) {
  const G4TouchableHandle& thisTouchable(aTrack.GetTouchableHandle());

  // Initialize parameters
  aParticleChange.Initialize(aTrack);
  const G4DynamicParticle* incomingRhadron = aTrack.GetDynamicParticle();
  CustomParticle* customIncomingRhadron =
      static_cast<CustomParticle*>(incomingRhadron->GetDefinition());  // This is used to get the cloud particle
  const G4ThreeVector& aPosition = aTrack.GetPosition();               // Position of the track
  const G4int incomingRhadronPDG = incomingRhadron->GetDefinition()->GetPDGEncoding();
  G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
  G4bool incomingRhadronSurvives = false;
  G4bool TargetSurvives = false;
  G4Nucleus targetNucleus(aTrack.GetMaterial());

  G4ParticleDefinition* outgoingRhadronDefinition = nullptr;
  G4ParticleDefinition* outgoingCloudDefinition = nullptr;
  G4ParticleDefinition* outgoingTargetDefinition = nullptr;

  // Declare the quark cloud as a G4DynamicParticle
  G4DynamicParticle* cloudParticle = new G4DynamicParticle();
  cloudParticle->SetDefinition(customIncomingRhadron->GetCloud());

  // Define the gluino and quark cloud G4LorentzVector (momentum, total energy) based on the momentum of the R-hadron and the ratio of the masses
  double scale = cloudParticle->GetDefinition()->GetPDGMass() / incomingRhadron->GetDefinition()->GetPDGMass();
  G4LorentzVector cloudMomentum(
      incomingRhadron->GetMomentum() * scale,
      std::sqrt(incomingRhadron->GetMomentum().mag() * scale * incomingRhadron->GetMomentum().mag() * scale +
                cloudParticle->GetDefinition()->GetPDGMass() * cloudParticle->GetDefinition()->GetPDGMass()));
  cloudParticle->Set4Momentum(cloudMomentum);
  G4LorentzVector gluinoMomentum(incomingRhadron->GetMomentum() * (1. - scale),
                                 incomingRhadron->GetTotalEnergy() - cloudParticle->GetTotalEnergy());

  // Update the cloud kinetic energy based on the target nucleus and evaporative effects
  G4double cloudKineticEnergy = cloudParticle->GetKineticEnergy();
  G4double initialCloudEnergy = cloudParticle->GetTotalEnergy();
  cloudKineticEnergy += targetNucleus.Cinema(cloudKineticEnergy);
  cloudKineticEnergy -= targetNucleus.EvaporationEffects(cloudKineticEnergy);

  G4ThreeVector cloud3MomentumDirection = cloudParticle->GetMomentum().unit();
  G4double cloud3MomentumMagnitudeAfterEvaporativeEffects =
      std::sqrt(cloudKineticEnergy * (cloudKineticEnergy + 2. * cloudParticle->GetDefinition()->GetPDGMass()));

  // If the R-hadron kinetic energy is less than 0.1 MeV, or the cloud kinetic energy is less than or equal to 0, stop the track but keep it alive. This should be very rare.
  if (cloudKineticEnergy + gluinoMomentum.e() - gluinoMomentum.m() <= 0.1 * MeV || cloudKineticEnergy <= 0.) {
    aParticleChange.ProposeTrackStatus(fStopButAlive);
    return &aParticleChange;
  }

  cloudParticle->SetKineticEnergy(cloudKineticEnergy);
  cloudParticle->SetMomentum(cloud3MomentumMagnitudeAfterEvaporativeEffects * cloud3MomentumDirection);

  //Get the final state particles. reactionProduct is a vector of integer PDGIDs. Not to be confused with G4ReactionProduct
  G4ParticleDefinition* aTarget;
  ReactionProduct reactionProduct = theHelper->GetFinalState(aTrack, aTarget);
  G4int reactionProductSize = reactionProduct.size();

  //Process outgoing particles from reactions
  std::vector<G4ParticleDefinition*> outgoingParticleDefinitions;
  for (ReactionProduct::iterator it = reactionProduct.begin(); it != reactionProduct.end(); ++it) {
    G4ParticleDefinition* finalStateParticle = theParticleTable->FindParticle(*it);
    CustomParticle* finalStateCustomParticle = dynamic_cast<CustomParticle*>(finalStateParticle);

    if (finalStateParticle == aTarget) {
      TargetSurvives = true;
    }

    if (finalStateParticle->GetParticleType() == "rhadron" || finalStateParticle->GetParticleType() == "mesonino" ||
        finalStateParticle->GetParticleType() == "sbaryon") {
      outgoingRhadronDefinition = finalStateParticle;
      outgoingCloudDefinition = finalStateCustomParticle->GetCloud();
    }

    if (finalStateParticle == G4Proton::Proton() || finalStateParticle == G4Neutron::Neutron())
      outgoingTargetDefinition = finalStateParticle;
    if (finalStateCustomParticle == nullptr && reactionProduct.size() == 2)
      outgoingTargetDefinition = finalStateParticle;
    if (finalStateParticle->GetPDGEncoding() == incomingRhadronPDG) {
      incomingRhadronSurvives = true;
    } else {
      outgoingParticleDefinitions.push_back(finalStateParticle);
    }
  }

  //If no reaction occured, set the outgoingTargetDefinition to the original target definition
  if (outgoingTargetDefinition == nullptr)
    outgoingTargetDefinition = theParticleTable->FindParticle(reactionProduct[1]);

  //If the incident particle survives, decrement the number of secondaries
  if (incomingRhadronSurvives)
    reactionProductSize--;
  aParticleChange.SetNumberOfSecondaries(reactionProductSize);

  //Create G4DynamicParticle and G4ReactionProduct objects for the outgoing target particle
  G4DynamicParticle* outgoingTargetG4Dynamic = new G4DynamicParticle;
  G4ReactionProduct outgoingTargetG4Reaction;
  if (TargetSurvives) {
    outgoingTargetG4Dynamic->SetDefinition(aTarget);
    outgoingTargetG4Reaction = G4ReactionProduct(aTarget);
  } else {
    outgoingTargetG4Dynamic->SetDefinition(outgoingTargetDefinition);
    outgoingTargetG4Reaction = G4ReactionProduct(outgoingTargetDefinition);
  }

  //Calculate the Lorentz boost of the cloud particle to the lab frame
  G4HadProjectile* incomingCloudG4HadProjectile = new G4HadProjectile(*cloudParticle);
  G4LorentzRotation cloudParticleToLabFrameRotation = incomingCloudG4HadProjectile->GetTrafoToLab();

  //Create a G4ReactionProduct object for the outgoing cloud
  G4ReactionProduct outgoingCloudG4Reaction(
      const_cast<G4ParticleDefinition*>(incomingCloudG4HadProjectile->GetDefinition()));
  outgoingCloudG4Reaction.SetMomentum(incomingCloudG4HadProjectile->Get4Momentum().vect());
  outgoingCloudG4Reaction.SetTotalEnergy(incomingCloudG4HadProjectile->GetTotalEnergy());
  if (!incomingRhadronSurvives) {
    outgoingCloudG4Reaction.SetDefinitionAndUpdateE(outgoingCloudDefinition);
  }

  G4ReactionProduct modifiedoutgoingCloudG4Reaction = outgoingCloudG4Reaction;

  //Set the hemisphere of the current and target particles. Initialize an empty vector for the secondary particles
  outgoingCloudG4Reaction.SetSide(1);    // incident always goes in forward hemisphere
  outgoingTargetG4Reaction.SetSide(-1);  // target always goes in backward hemisphere
  G4bool quasiElastic = false;
  if (reactionProduct.size() == 2)
    quasiElastic = true;
  G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> secondaryParticleVector;
  G4int secondaryParticleVectorLen = 0;
  secondaryParticleVector.Initialize(0);

  //Fill the vector with the secondary particles
  for (G4int i = 0; i != reactionProductSize; i++) {
    if (outgoingParticleDefinitions[i] != aTarget &&
        outgoingParticleDefinitions[i] != incomingCloudG4HadProjectile->GetDefinition() &&
        outgoingParticleDefinitions[i] != outgoingRhadronDefinition &&
        outgoingParticleDefinitions[i] != outgoingTargetDefinition) {
      G4ReactionProduct* secondaryReactionProduct = new G4ReactionProduct;
      secondaryReactionProduct->SetDefinition(outgoingParticleDefinitions[i]);
      (G4UniformRand() < 0.5)
          ? secondaryReactionProduct->SetSide(-1)
          : secondaryReactionProduct->SetSide(1);  //Here we randomly determine the hemisphere of the secondary particle
      secondaryParticleVector.SetElement(secondaryParticleVectorLen++, secondaryReactionProduct);
    }
  }

  //Store the outgoing Cloud 4-momentum for energy deposit calculation that occurs after change in momemta has been calculated
  G4LorentzVector outgoingCloudp4(outgoingCloudG4Reaction.GetMomentum(), outgoingCloudG4Reaction.GetTotalEnergy());
  outgoingCloudp4 *= cloudParticleToLabFrameRotation;

  G4bool incomingRhadronHasChanged = !incomingRhadronSurvives;
  G4bool targetHasChanged = !TargetSurvives;
  CalculateMomenta(secondaryParticleVector,
                   secondaryParticleVectorLen,
                   incomingCloudG4HadProjectile,
                   outgoingTargetG4Dynamic,
                   modifiedoutgoingCloudG4Reaction,
                   targetNucleus,
                   outgoingCloudG4Reaction,
                   outgoingTargetG4Reaction,
                   incomingRhadronHasChanged,
                   targetHasChanged,
                   quasiElastic);

  //Declare the Cloud 4-momentum after the interaction and propose an energy deposit of the difference between the incoming and outgoing quark cloud energies
  G4LorentzVector outgoingCloudp4Prime(outgoingCloudG4Reaction.GetMomentum(), outgoingCloudG4Reaction.GetTotalEnergy());
  outgoingCloudp4Prime *= cloudParticleToLabFrameRotation;
  G4double proposedEnergyDeposit = initialCloudEnergy - outgoingCloudG4Reaction.GetTotalEnergy();

  if (proposedEnergyDeposit > 0) {
    aParticleChange.ProposeLocalEnergyDeposit(proposedEnergyDeposit);
  }

  //Update the number of secondaries to the correct value
  aParticleChange.SetNumberOfSecondaries(secondaryParticleVectorLen + reactionProductSize);

  //If the incident particle does not survive, update the outgoing track to be the new R-Hadron with the proper momentum, time, and position
  G4DynamicParticle* dynamicOutgoingRhadron = new G4DynamicParticle;
  if (!incomingRhadronSurvives) {
    dynamicOutgoingRhadron->SetDefinition(outgoingRhadronDefinition);
    dynamicOutgoingRhadron->SetMomentum(gluinoMomentum.vect() + outgoingCloudp4Prime.vect());

    G4Track* outgoingRhadronTrack = new G4Track(dynamicOutgoingRhadron, aTrack.GetGlobalTime(), aPosition);
    outgoingRhadronTrack->SetTouchableHandle(thisTouchable);
    aParticleChange.AddSecondary(outgoingRhadronTrack);

    //Stop the old track
    aParticleChange.ProposeTrackStatus(fStopAndKill);
  }

  //If the incident particle survives update its momentum direction. Includes error handling for when the momentum is zero
  else {
    dynamicOutgoingRhadron->SetDefinition(incomingRhadron->GetDefinition());
    dynamicOutgoingRhadron->SetMomentum(gluinoMomentum.vect() + outgoingCloudp4Prime.vect());
    if (dynamicOutgoingRhadron->GetMomentum().mag() > DBL_MIN)
      aParticleChange.ProposeMomentumDirection(dynamicOutgoingRhadron->GetMomentumDirection());
    else
      aParticleChange.ProposeMomentumDirection(1.0, 0.0, 0.0);
    aParticleChange.ProposeEnergy(dynamicOutgoingRhadron->GetKineticEnergy());
  }

  //Update the momenta of the target track
  G4DynamicParticle* targetParticleG4DynamicAfterInteraction = new G4DynamicParticle;
  if (outgoingTargetG4Reaction.GetMass() > 0.0)  // outgoingTargetG4Reaction can be eliminated in TwoBody
  {
    targetParticleG4DynamicAfterInteraction->SetDefinition(outgoingTargetG4Reaction.GetDefinition());
    targetParticleG4DynamicAfterInteraction->SetMomentum(outgoingTargetG4Reaction.GetMomentum().rotate(
        2. * pi * G4UniformRand(),
        incomingCloud3Momentum));  // rotate(const G4double angle, const ThreeVector &axis) const;
    targetParticleG4DynamicAfterInteraction->SetMomentum(
        (cloudParticleToLabFrameRotation * targetParticleG4DynamicAfterInteraction->Get4Momentum()).vect());
    G4Track* targetTrackAfterInteraction =
        new G4Track(targetParticleG4DynamicAfterInteraction, aTrack.GetGlobalTime(), aPosition);
    targetTrackAfterInteraction->SetTouchableHandle(thisTouchable);
    aParticleChange.AddSecondary(targetTrackAfterInteraction);
  }

  // Update the momenta of the remaining secondary tracks
  for (int i = 0; i < secondaryParticleVectorLen; ++i) {
    G4DynamicParticle* secondaryParticleAfterInteraction = new G4DynamicParticle();
    secondaryParticleAfterInteraction->SetDefinition(secondaryParticleVector[i]->GetDefinition());
    secondaryParticleAfterInteraction->SetMomentum(secondaryParticleVector[i]->GetMomentum());
    secondaryParticleAfterInteraction->SetMomentum(
        (cloudParticleToLabFrameRotation * secondaryParticleAfterInteraction->Get4Momentum()).vect());
    G4Track* secondaryTrackAfterInteraction =
        new G4Track(secondaryParticleAfterInteraction, aTrack.GetGlobalTime(), aPosition);
    secondaryTrackAfterInteraction->SetTouchableHandle(thisTouchable);
    aParticleChange.AddSecondary(secondaryTrackAfterInteraction);

    delete secondaryParticleVector[i];
  }

  delete incomingCloudG4HadProjectile;
  delete outgoingTargetG4Dynamic;
  //aParticleChange.DumpInfo();
  ClearNumberOfInteractionLengthLeft();

  return &aParticleChange;
}

void FullModelHadronicProcess::CalculateMomenta(
    G4FastVector<G4ReactionProduct, MYGHADLISTSIZE>& secondaryParticleVector,  //Vector of secondary particles
    G4int& secondaryParticleVectorLen,                                         //Length of the secondary particle vector
    const G4HadProjectile* incomingCloudG4HadProjectile,                       //The incoming cloud projectile
    const G4DynamicParticle* outgoingTargetG4Dynamic,                          //The original target particle
    G4ReactionProduct& modifiedoutgoingCloudG4Reaction,                        //Fermi motion and evap. effects included
    G4Nucleus& targetNucleus,                                                  //The target nucleus
    G4ReactionProduct& outgoingCloudG4Reaction,                                //The outgoing cloud G4 Reaction
    G4ReactionProduct& outgoingTargetG4Reaction,  //The outgoing particle previously defined as original target
    G4bool& incomingRhadronHasChanged,            //True if the R-Hadron type has changed
    G4bool& targetHasChanged,                     //True if the target particle has changed
    G4bool quasiElastic)                          //True if the reaction product size equals 2, false otherwise
{
  FullModelReactionDynamics theReactionDynamics;
  incomingCloud3Momentum = incomingCloudG4HadProjectile->Get4Momentum().v();  //Use this for rotations later

  //If the reaction is quasi-elastic, use the TwoBody method to calculate the momenta of the outgoing particles.
  if (quasiElastic) {
    theReactionDynamics.TwoBody(secondaryParticleVector,
                                secondaryParticleVectorLen,
                                modifiedoutgoingCloudG4Reaction,
                                outgoingTargetG4Dynamic,
                                outgoingCloudG4Reaction,
                                outgoingTargetG4Reaction,
                                targetNucleus,
                                targetHasChanged);
    return;
  }

  //If the reaction is not quasi-elastic, update the outgoing particles momenta based on effects detailed in the functions below. Then call the TwoBody method afterwards
  G4ReactionProduct leadingStrangeParticle;
  G4bool leadFlag =
      MarkLeadingStrangeParticle(outgoingCloudG4Reaction, outgoingTargetG4Reaction, leadingStrangeParticle);
  G4bool finishedTwoClu = false;
  if (modifiedoutgoingCloudG4Reaction.GetTotalMomentum() / MeV < 1.0) {
    for (G4int i = 0; i < secondaryParticleVectorLen; i++) {
      delete secondaryParticleVector[i];
    }
    secondaryParticleVectorLen = 0;
  } else {
    theReactionDynamics.SuppressChargedPions(secondaryParticleVector,
                                             secondaryParticleVectorLen,
                                             modifiedoutgoingCloudG4Reaction,
                                             outgoingCloudG4Reaction,
                                             outgoingTargetG4Reaction,
                                             targetNucleus,
                                             incomingRhadronHasChanged,
                                             targetHasChanged);

    try {
      finishedTwoClu = theReactionDynamics.TwoCluster(secondaryParticleVector,
                                                      secondaryParticleVectorLen,
                                                      modifiedoutgoingCloudG4Reaction,
                                                      incomingCloudG4HadProjectile,
                                                      outgoingCloudG4Reaction,
                                                      outgoingTargetG4Reaction,
                                                      targetNucleus,
                                                      incomingRhadronHasChanged,
                                                      targetHasChanged,
                                                      leadFlag,
                                                      leadingStrangeParticle);
    } catch (G4HadronicException& aR) {
      G4ExceptionDescription ed;
      aR.Report(ed);
      G4Exception("FullModelHadronicProcess::CalculateMomenta", "had066", FatalException, ed);
    }
  }

  if (finishedTwoClu) {
    Rotate(secondaryParticleVector, secondaryParticleVectorLen);
    return;
  }

  theReactionDynamics.TwoBody(secondaryParticleVector,
                              secondaryParticleVectorLen,
                              modifiedoutgoingCloudG4Reaction,
                              outgoingTargetG4Dynamic,
                              outgoingCloudG4Reaction,
                              outgoingTargetG4Reaction,
                              targetNucleus,
                              targetHasChanged);
}

G4bool FullModelHadronicProcess::MarkLeadingStrangeParticle(const G4ReactionProduct& outgoingCloudG4Reaction,
                                                            const G4ReactionProduct& outgoingTargetG4Reaction,
                                                            G4ReactionProduct& leadParticle) {
  //Here we check to see if the current or target particle is more massive than the Kaon, not a proton, and not a neutron. If so, we set the lead particle to the strange particle
  G4bool lead = false;
  if ((outgoingCloudG4Reaction.GetMass() >= G4KaonPlus::KaonPlus()->GetPDGMass()) &&
      (outgoingCloudG4Reaction.GetDefinition() != G4Proton::Proton()) &&
      (outgoingCloudG4Reaction.GetDefinition() != G4Neutron::Neutron())) {
    lead = true;
    leadParticle = outgoingCloudG4Reaction;
  } else if ((outgoingTargetG4Reaction.GetMass() >= G4KaonPlus::KaonPlus()->GetPDGMass()) &&
             (outgoingTargetG4Reaction.GetDefinition() != G4Proton::Proton()) &&
             (outgoingTargetG4Reaction.GetDefinition() != G4Neutron::Neutron())) {
    lead = true;
    leadParticle = outgoingTargetG4Reaction;
  }
  return lead;
}

void FullModelHadronicProcess::Rotate(G4FastVector<G4ReactionProduct, MYGHADLISTSIZE>& secondaryParticleVector,
                                      G4int& secondaryParticleVectorLen) {
  G4int i;
  for (i = 0; i < secondaryParticleVectorLen; ++i) {
    G4ThreeVector momentum = secondaryParticleVector[i]->GetMomentum();
    momentum = momentum.rotate(2. * pi * G4UniformRand(), incomingCloud3Momentum);
    secondaryParticleVector[i]->SetMomentum(momentum);
  }
}