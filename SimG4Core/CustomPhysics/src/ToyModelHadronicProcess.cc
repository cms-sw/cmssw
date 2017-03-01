//Geant4 include
#include "G4ProcessManager.hh"
#include "G4ParticleTable.hh"
#include "G4Track.hh"
#include "G4FermiPhaseSpaceDecay.hh"
//Our includes
#include "SimG4Core/CustomPhysics/interface/ToyModelHadronicProcess.hh"
#include "SimG4Core/CustomPhysics/interface/CustomPDGParser.h"
#include "SimG4Core/CustomPhysics/interface/CustomParticle.h"
#include "SimG4Core/CustomPhysics/interface/HadronicProcessHelper.hh"
#include "SimG4Core/CustomPhysics/interface/Decay3Body.h"

#include <vector>

using namespace CLHEP;

ToyModelHadronicProcess::ToyModelHadronicProcess(HadronicProcessHelper * aHelper, 
						 const G4String& processName) :
  G4VDiscreteProcess(processName), m_verboseLevel(0),  m_helper(aHelper),  m_detachCloud(true)
{
  m_decay = new G4FermiPhaseSpaceDecay();
}
 
G4bool ToyModelHadronicProcess::IsApplicable(const G4ParticleDefinition& aP)
{
  return m_helper->applicabilityTester(aP);
}

G4double ToyModelHadronicProcess::GetMicroscopicCrossSection(const G4DynamicParticle *particle,
							     const G4Element *element,
							     G4double /*temperature*/)
{
  //Get the cross section for this particle/element combination from the ProcessHelper
  G4double inclusiveCrossSection = m_helper->inclusiveCrossSection(particle,element);
   
  if(m_verboseLevel >= 3) {
    G4cout << "ToyModelHadronicProcess: Return cross section " << inclusiveCrossSection 
	   << G4endl;
  }
  return inclusiveCrossSection;
}

G4double ToyModelHadronicProcess::GetMeanFreePath(const G4Track& aTrack, G4double, 
						  G4ForceCondition*)
{
  G4Material *aMaterial = aTrack.GetMaterial();
  const G4DynamicParticle *aParticle = aTrack.GetDynamicParticle();
  G4double sigma = 0.0;

  G4int nElements = aMaterial->GetNumberOfElements();
  
  const G4double *theAtomicNumDensityVector =
    aMaterial->GetAtomicNumDensityVector();
  G4double aTemp = aMaterial->GetTemperature();
  
  for( G4int i=0; i<nElements; ++i )
    {
      G4double xSection =
	GetMicroscopicCrossSection( aParticle, (*aMaterial->GetElementVector())[i], aTemp);
      sigma += theAtomicNumDensityVector[i] * xSection;
    }
  G4double x = DBL_MAX;
  if(sigma > 0.0) { x = 1./sigma; }
  return x;  
}

G4VParticleChange* ToyModelHadronicProcess::PostStepDoIt(const G4Track& track,
							 const G4Step& /*  step*/)
{
  const G4TouchableHandle thisTouchable(track.GetTouchableHandle());
  
  // A little setting up
  G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
  m_particleChange.Initialize(track);

  //This will contain RHadron Def + RHad momentum
  const G4DynamicParticle* incidentRHadron = track.GetDynamicParticle(); 
  const G4ThreeVector aPosition = track.GetPosition();

  CustomParticle* CustomIncident = dynamic_cast<CustomParticle*>(incidentRHadron->GetDefinition());

  //This will contain Cloud Def + scaled momentum
  G4DynamicParticle* cloudParticle =  new G4DynamicParticle(); 
           
  //set cloud definition
  if(CustomIncident==0) {
    G4cout << "ToyModelHadronicProcess::PostStepDoIt  Definition of particle cloud not available!" 
	   << G4endl;
  } else {
    cloudParticle->SetDefinition(CustomIncident->GetCloud());
  }
     
  //compute scaled momentum  
  double scale = cloudParticle->GetDefinition()->GetPDGMass()
    /incidentRHadron->GetDefinition()->GetPDGMass();
  G4LorentzVector cloudMomentum(incidentRHadron->GetMomentum()*scale,
				cloudParticle->GetDefinition()->GetPDGMass());
  G4LorentzVector gluinoMomentum(incidentRHadron->GetMomentum()*(1.-scale),
				 CustomIncident->GetSpectator()->GetPDGMass());
  
  cloudParticle->Set4Momentum(cloudMomentum);         	
  
  const G4DynamicParticle *incidentParticle;
  if(! m_detachCloud) { incidentParticle = incidentRHadron; } 
  else                { incidentParticle = cloudParticle; }
  
  if(m_verboseLevel >= 3)
    { 
      G4cout << "ToyModelHadronicProcess::PostStepDoIt    After scaling " << G4endl;
      G4cout << "      RHadron "<<incidentRHadron->GetDefinition()->GetParticleName()
	     <<"         pdgmass= " << incidentRHadron->GetDefinition()->GetPDGMass() / GeV
	     << " P= " << incidentRHadron->Get4Momentum() / GeV<<" mass= "
	     << incidentRHadron->Get4Momentum().m() / GeV <<G4endl;
      G4cout << "      Cloud   "<<cloudParticle->GetDefinition()->GetParticleName()
	     <<"         pdgmass= " << cloudParticle->GetDefinition()->GetPDGMass() / GeV
	     << " P= " << cloudParticle->Get4Momentum() / GeV<<" mass= "
	     << cloudParticle->Get4Momentum().m() / GeV <<G4endl;
      G4cout << "      Incident          pdgmass= " 
	     << incidentParticle->GetDefinition()->GetPDGMass() / GeV
	     << " P= " << incidentParticle->Get4Momentum() / GeV<<" mass= "
	     << incidentParticle->Get4Momentum().m() / GeV <<G4endl;
    }
 
  const G4ThreeVector position = track.GetPosition();
  std::vector<G4ParticleDefinition*> newParticles;        // this will contain clouds
  std::vector<G4ParticleDefinition*> newParticlesRHadron; // this will contain r-hadron

  //Get the final state particles and target
  G4ParticleDefinition* targetParticle; 
  HadronicProcessHelper::ReactionProduct reactionProduct = 
    m_helper->finalState(incidentRHadron,track.GetMaterial(),targetParticle);

  // Fill a list of the new particles to create including the incident if it survives
  for(HadronicProcessHelper::ReactionProduct::iterator it  = reactionProduct.begin();
      it != reactionProduct.end() ;  
      ++it)
    {
      G4ParticleDefinition * productDefinition =theParticleTable->FindParticle(*it);
      newParticlesRHadron.push_back(productDefinition);
      CustomParticle* cProd = dynamic_cast<CustomParticle*>(productDefinition);

      if( cProd!=0 && m_detachCloud)
	productDefinition=cProd->GetCloud();

      newParticles.push_back(productDefinition);
    }
  
  int numberOfSecondaries = reactionProduct.size();
  
  if(m_verboseLevel >= 2) {
    G4cout << "ToyModelHadronicProcess::PostStepDoIt  N secondaries: " 
	   << numberOfSecondaries << G4endl;
  }

  // Getting 4-momenta
  G4LorentzVector sum4Momentum = incidentParticle->Get4Momentum();
  G4LorentzVector target4Momentum(0,0,0,targetParticle->GetPDGMass());
  sum4Momentum += target4Momentum;
  G4ThreeVector   cmBoost = sum4Momentum.boostVector();
  G4double M = sum4Momentum.m();

  if(m_verboseLevel >= 2) {
    G4cout << "ToyModelHadronicProcess::PostStepDoIt  Kinematics in GeV: " << G4endl 
	   << "     Boost   = " << cmBoost / GeV<< G4endl
	   << "     4P Lab  = " << sum4Momentum / GeV << G4endl
	   << "     Ecm  = " << M / GeV << G4endl;
  }
  std::vector<G4double> m;

  //Fill the masses
  for(int ip=0; ip<numberOfSecondaries; ++ip)     
    {
      m.push_back(newParticles[ip]->GetPDGMass());
    }

  std::vector<G4LorentzVector*>* fourMomenta = m_decay->Decay(M, m);

  if(fourMomenta) {
 
    //incident particle has to be killed 
    m_particleChange.SetNumberOfSecondaries(numberOfSecondaries);
    m_particleChange.ProposeTrackStatus(fStopAndKill);
    if(m_verboseLevel >= 3) {
      G4cout  << "Incident does not survive: stopAndKill + set num secondaries to " 
	      << numberOfSecondaries << G4endl;
    }  

    for (int ip=0; ip <numberOfSecondaries; ++ip) {
      // transform to Lab
      (*fourMomenta)[ip]->boost(cmBoost);

      // added cloud to R-hadron
      if (newParticlesRHadron[ip]==incidentRHadron->GetDefinition()) {
	if(m_detachCloud) {
	  if(m_verboseLevel >= 3) {
	    G4cout  << "ToyModelHadronicProcess::PostStepDoIt   Add gluino momentum " 
		    << *((*fourMomenta)[ip])/GeV <<"(m="<< (*fourMomenta)[ip]->m()/ GeV
		    <<") + " <<gluinoMomentum/GeV<<G4endl; 
	  }
	  G4LorentzVector p4_new = gluinoMomentum + *((*fourMomenta)[ip]);
	  G4ThreeVector momentum = p4_new.vect();
	  double rhMass = newParticlesRHadron[ip]->GetPDGMass();
	  (*fourMomenta)[ip]->setVectM(momentum,rhMass);
	  if(m_verboseLevel >= 3) {
	    G4cout <<  " = " << *((*fourMomenta)[ip])/GeV <<"(m="<< (*fourMomenta)[ip]->m() / GeV
		   <<") vs. "<<rhMass/GeV 
		   << G4endl;     	
	  }
	}
      }
      //Create new dynamic particle
      G4DynamicParticle* productDynParticle = new G4DynamicParticle();
      //Set the pDef to the dynParte
      productDynParticle->SetDefinition(newParticlesRHadron[ip]);
      //Set the 4-vector to dynPart
      productDynParticle->Set4Momentum(*((*fourMomenta)[ip]));  

      if(m_verboseLevel >= 3) {
	G4cout  << "ToyModelHadronicProcess::PostStepDoIt: " << *((*fourMomenta)[ip])/GeV 
		<< "(m="<< (*fourMomenta)[ip]->m()/ GeV<<") " 
		<< newParticlesRHadron[ip]->GetParticleName()
		<< G4endl;     	
      }
      //Create a G4Track
      G4Track* productTrack = new G4Track(productDynParticle,
					  track.GetGlobalTime(),
					  position);
      productTrack->SetTouchableHandle(thisTouchable);
      m_particleChange.AddSecondary(productTrack);
      delete (*fourMomenta)[ip];
    }
    delete fourMomenta;
  } 
  //clear interaction length      
  ClearNumberOfInteractionLengthLeft();
  return &m_particleChange;
}

