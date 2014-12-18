#include "G4HadReentrentException.hh"
#include "G4ProcessManager.hh"
#include "G4ParticleTable.hh"

#include "SimG4Core/CustomPhysics/interface/FullModelHadronicProcess.hh"
#include "SimG4Core/CustomPhysics/interface/G4ProcessHelper.hh"
#include "SimG4Core/CustomPhysics/interface/Decay3Body.h"
#include "SimG4Core/CustomPhysics/interface/CustomPDGParser.h"
#include "SimG4Core/CustomPhysics/interface/CustomParticle.h"

using namespace CLHEP;

FullModelHadronicProcess::FullModelHadronicProcess(G4ProcessHelper * aHelper, const G4String& processName) :
  G4VDiscreteProcess(processName), theHelper(aHelper)
{}


FullModelHadronicProcess::~FullModelHadronicProcess(){}
  
G4bool FullModelHadronicProcess::IsApplicable(const G4ParticleDefinition& aP)
{
  return theHelper->ApplicabilityTester(aP);  
}

G4double FullModelHadronicProcess::GetMicroscopicCrossSection(const G4DynamicParticle *aParticle,
							      const G4Element *anElement,
							      G4double aTemp)
{
  //Get the cross section for this particle/element combination from the ProcessHelper
  G4double InclXsec = theHelper->GetInclusiveCrossSection(aParticle,anElement);
  //  G4cout<<"Returned cross section from helper was: "<<InclXsec/millibarn<<" millibarn"<<G4endl;

  // Need to provide Set-methods for these in time
  G4double HighestEnergyLimit = 10 * TeV  ;
  G4double LowestEnergyLimit = 1 * eV;

  G4double ParticleEnergy = aParticle->GetKineticEnergy();

   
  if (ParticleEnergy > HighestEnergyLimit || ParticleEnergy < LowestEnergyLimit){
    return 0;
  } else {
    //    G4cout << "Microscopic Cross Section: "<<InclXsec / millibarn<<" millibarn"<<G4endl;
    return InclXsec;
  }

}

G4double FullModelHadronicProcess::GetMeanFreePath(const G4Track& aTrack, G4double, G4ForceCondition*)
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

  return 1.0/sigma;
  
}

G4VParticleChange* FullModelHadronicProcess::PostStepDoIt(const G4Track& aTrack,
							  const G4Step&  aStep)
{
  //  G4cout<<"*****************    Entering FullModelHadronicProcess::PostStepDoIt       **********************"<<G4endl;

  const G4TouchableHandle thisTouchable(aTrack.GetTouchableHandle());

  // A little setting up
  aParticleChange.Initialize(aTrack);
  //  G4DynamicParticle* OrgPart = const_cast<G4DynamicParticle*>(aTrack.GetDynamicParticle());
  G4DynamicParticle* IncidentRhadron = const_cast<G4DynamicParticle*>(aTrack.GetDynamicParticle());
  CustomParticle* CustomIncident = static_cast<CustomParticle*>(IncidentRhadron->GetDefinition());
  const G4ThreeVector aPosition = aTrack.GetPosition();
  //  std::cout<<"G: "<<aStep.GetStepLength()/cm<<std::endl;
  const G4int theIncidentPDG = IncidentRhadron->GetDefinition()->GetPDGEncoding();
  G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
  std::vector<G4ParticleDefinition*> theParticleDefinitions;
  //  std::vector<G4DynamicParticle*> *theDynamicParticles;//These are probably redundant, but they can easily be removed :-)
  G4bool IncidentSurvives = false;
  G4bool TargetSurvives = false;
  G4Nucleus targetNucleus(aTrack.GetMaterial());
  G4ParticleDefinition* outgoingRhadron=0;
  G4ParticleDefinition* outgoingCloud=0;
  G4ParticleDefinition* outgoingTarget=0;
//  double gamma = IncidentRhadron->GetTotalEnergy()/IncidentRhadron->GetDefinition()->GetPDGMass();


  G4ThreeVector p_0 = IncidentRhadron->GetMomentum();

  //  static int n=0;

  G4double e_kin_0 = IncidentRhadron->GetKineticEnergy();
  //  G4cout<<e_kin_0/GeV<<G4endl;

  G4DynamicParticle* cloudParticle = new G4DynamicParticle();
  /*
  if(CustomPDGParser::s_isRMeson(theIncidentPDG))
    std::cout<<"Rmeson"<<std::endl;
  if(CustomPDGParser::s_isRBaryon(theIncidentPDG)) 
    std::cout<<"Rbaryon"<<std::endl;
  */
  /*
  if(CustomPDGParser::s_isRBaryon(theIncidentPDG)) 
     cloudParticle->SetDefinition(theParticleTable->FindParticle("rhadronbaryoncloud"));
  if(CustomPDGParser::s_isRMeson(theIncidentPDG) || CustomPDGParser::s_isRGlueball(theIncidentPDG) )
     cloudParticle->SetDefinition(theParticleTable->FindParticle("rhadronmesoncloud"));
  */
  cloudParticle->SetDefinition(CustomIncident->GetCloud());

  if(cloudParticle->GetDefinition() == 0) 
     {
      std::cout << "FullModelHadronicProcess::PostStepDoIt  Definition of particle cloud not available!!" << std::endl;
     }
  /*
  std::cout<<"Incoming particle was "<<IncidentRhadron->GetDefinition()->GetParticleName()<<". Corresponding cloud is "<<cloudParticle->GetDefinition()->GetParticleName()<<std::endl;
  G4cout<<"Kinetic energy was: "<<IncidentRhadron->GetKineticEnergy()/GeV<<" GeV"<<G4endl;
  */
  double scale=cloudParticle->GetDefinition()->GetPDGMass()/IncidentRhadron->GetDefinition()->GetPDGMass();
  //  std::cout<<"Mass ratio: "<<scale<<std::endl;
  G4LorentzVector cloudMomentum;
  cloudMomentum.setVectM(IncidentRhadron->GetMomentum()*scale,cloudParticle->GetDefinition()->GetPDGMass());
  G4LorentzVector gluinoMomentum;
  //  gluinoMomentum.setVectM(IncidentRhadron->GetMomentum()*(1.-scale),theParticleTable->FindParticle("~g")->GetPDGMass()); 
  gluinoMomentum.setVectM(IncidentRhadron->GetMomentum()*(1.-scale),CustomIncident->GetSpectator()->GetPDGMass()); 
  /*
  G4cout <<"Are these the same?"<<G4endl;
  G4cout<<gluinoMomentum<<G4endl;
  G4cout<<(1.-scale) * IncidentRhadron->Get4Momentum()<<G4endl;
  */
  //These two for getting CMS transforms later (histogramming purposes...)
  G4LorentzVector FullRhadron4Momentum = IncidentRhadron->Get4Momentum();
  G4LorentzVector Cloud4Momentum = cloudMomentum;

  cloudParticle->Set4Momentum(cloudMomentum);         	

  G4DynamicParticle* OrgPart = cloudParticle;

  /*
  std::cout<<"Original momentum: "<<IncidentRhadron->Get4Momentum().v().mag()/GeV<<" GeV, corresponding to gamma: "
  	   <<IncidentRhadron->GetTotalEnergy()/IncidentRhadron->GetDefinition()->GetPDGMass()<<std::endl;
  
  std::cout<<"Cloud momentum: "<<cloudParticle->Get4Momentum().v().mag()/GeV<<" GeV, corresponding to gamma: "
	   <<cloudParticle->GetTotalEnergy()/cloudParticle->GetDefinition()->GetPDGMass()<<std::endl;
  */

  double E_0 = IncidentRhadron->GetKineticEnergy();


  G4double ek = OrgPart->GetKineticEnergy();
  G4double amas = OrgPart->GetDefinition()->GetPDGMass();
  
  G4double tkin = targetNucleus.Cinema( ek );
  ek += tkin;
  OrgPart->SetKineticEnergy( ek );
  G4double et = ek + amas;
  G4double p = std::sqrt( std::abs((et-amas)*(et+amas)) );
  G4double pp = OrgPart->GetMomentum().mag();
  if( pp > 0.0 )
    {
      G4ThreeVector momentum = OrgPart->GetMomentum();
      OrgPart->SetMomentum( momentum * (p/pp) );
    }
  
  // calculate black track energies
  if(ek > 0.) {  tkin = targetNucleus.EvaporationEffects( ek );  ek -= tkin; } // AR_NEWCODE_IMPORT
  if(ek+gluinoMomentum.e()-gluinoMomentum.m()<=0.1*MeV||ek<=0.) {
    //Very rare event...
    G4cout<<"Kinetic energy is sick"<<G4endl;
    G4cout<<"Full R-hadron: "<<(ek+gluinoMomentum.e()-gluinoMomentum.m())/MeV<<" MeV" <<G4endl;
    G4cout<<"Quark system: "<<ek/MeV<<" MeV"<<G4endl;
//    aParticleChange.ProposeTrackStatus( fStopAndKill ); // AR_NEWCODE_IMPORT
    aParticleChange.ProposeTrackStatus( fStopButAlive ); // AR_NEWCODE_IMPORT
    return &aParticleChange;
  }
  OrgPart->SetKineticEnergy( ek );
  et = ek + amas;
  p = std::sqrt( std::abs((et-amas)*(et+amas)) );
  pp = OrgPart->GetMomentum().mag();
  
  if( pp > 0.0 )
    {
      G4ThreeVector momentum = OrgPart->GetMomentum();
      OrgPart->SetMomentum( momentum * (p/pp) );
    }



  //Get the final state particles
  
  G4ParticleDefinition* aTarget; 
  ReactionProduct rp = theHelper->GetFinalState(aTrack,aTarget);
  G4bool force2to2 = false;
  //  G4cout<<"Trying to get final state..."<<G4endl; 
  while(rp.size()!=2 && force2to2){
    rp = theHelper->GetFinalState(aTrack,aTarget);
  }
  G4int NumberOfSecondaries = rp.size();
  //  G4cout<<"Multiplicity of selected final state: "<<rp.size()<<G4endl;

  //Getting CMS transforms. Boosting is done at histogram filling
  G4LorentzVector Target4Momentum;
  Target4Momentum.set(0.,0.,0.,aTarget->GetPDGMass());
  //  Target4Momentum.setVectM(0.,targetNucleus.GetN()*GeV);
  G4LorentzVector psum_full,psum_cloud;
  psum_full = FullRhadron4Momentum + Target4Momentum;
  psum_cloud = Cloud4Momentum + Target4Momentum;
  G4ThreeVector trafo_full_cms = (-1)*psum_full.boostVector();
  G4ThreeVector trafo_cloud_cms = (-1)*psum_cloud.boostVector();
  /*
  psum_full.boost(trafo_full_cms);
  psum_cloud.boost(trafo_cloud_cms);
  std::cout<<"Checking that the momenta are in deed zero:"<<psum_full.vect()<<std::endl;
  */

  // OK Let's make some particles :-)
  // We're not using them for anything yet, I know, but let's make sure the machinery is there

  for(ReactionProduct::iterator it  = rp.begin();
      it != rp.end();
      it++)
    {
      G4ParticleDefinition* tempDef = theParticleTable->FindParticle(*it);
      CustomParticle* tempCust = dynamic_cast<CustomParticle*>(tempDef);
      if (tempDef==aTarget) TargetSurvives = true;

      //      if (tempDef->GetParticleType()=="rhadron")
      if (tempCust!=0)
	{
	  outgoingRhadron = tempDef; 
	  //Setting outgoing cloud definition
	  /*
	  if(CustomPDGParser::s_isRBaryon(*it)) 
	    outgoingCloud=theParticleTable->FindParticle("rhadronbaryoncloud");
	  if(CustomPDGParser::s_isRMeson(*it) || CustomPDGParser::s_isRGlueball(*it) )
	    outgoingCloud=theParticleTable->FindParticle("rhadronmesoncloud");
	  */
	  outgoingCloud=tempCust->GetCloud();
	  if(outgoingCloud == 0) 
	    {
	      std::cout << "FullModelHadronicProcess::PostStepDoIt  Definition of outgoing particle cloud not available!!" << std::endl;
	    }
	  /*
	  std::cout<<"Outgoing Rhadron is: "<<outgoingRhadron->GetParticleName()<<std::endl;
	  std::cout<<"Outgoing cloud is: "<<outgoingCloud->GetParticleName()<<std::endl;
	  */
	}

      if (tempDef==G4Proton::Proton()||tempDef==G4Neutron::Neutron()) outgoingTarget = tempDef;
      //      if (tempDef->GetParticleType()!="rhadron"&&rp.size()==2) outgoingTarget = tempDef;
      if (tempCust==0&&rp.size()==2) outgoingTarget = tempDef;
      if (tempDef->GetPDGEncoding()==theIncidentPDG) {
	IncidentSurvives = true;
      } else {
	theParticleDefinitions.push_back(tempDef);
	/*
	G4DynamicParticle* tempDyn = new G4DynamicParticle();
	tempDyn->SetDefinition(tempDef);
	theDynamicParticles->push_back(tempDyn);
	*/
      }
    }

  //Not using this, so...
  //  delete theDynamicParticles;

  if (outgoingTarget==0) outgoingTarget = theParticleTable->FindParticle(rp[1]);

  // A little debug information
  /*
  G4cout<<"The particles coming out of this reaction will be: ";
  for (std::vector<G4DynamicParticle*>::iterator it = theDynamicParticles.begin();
       it != theDynamicParticles.end();
       it++){
    G4cout<< (*it)->GetDefinition()->GetParticleName()<<" ";
  }
  G4cout<<G4endl;
  */
  // If the incident particle survives it is not a "secondary", although
  // it would probably be easier to fStopAndKill the track every time.
  if(IncidentSurvives) NumberOfSecondaries--;
  aParticleChange.SetNumberOfSecondaries(NumberOfSecondaries);


  // ADAPTED FROM G4LEPionMinusInelastic::ApplyYourself
  // It is bloody ugly, but such is the way of cut 'n' paste


  // Set up the incident
  const G4HadProjectile* originalIncident = new G4HadProjectile(*OrgPart);//This is where rotation to z-axis is done (Aarrggh!)


  //Maybe I need to calculate trafo from R-hadron... Bollocks! Labframe does not move. CMS does.  
  G4HadProjectile* org2 = new G4HadProjectile(*OrgPart);
  G4LorentzRotation trans = org2->GetTrafoToLab();
  delete org2;
  
  //    if (originalIncident->GetKineticEnergy()<= 0.1*MeV) { //Needs rescaling. The kinetic energy of the quarksystem is the relevant quantity
  
  /*
  G4cout<<"Kinetic energies: "<<G4endl;
  G4cout<<"True kinetic energy:     "<<originalIncident->GetKineticEnergy()/GeV<<" GeV"<<G4endl;
  G4cout<<"Mass:                    "<<originalIncident->GetDefinition()->GetPDGMass()/GeV<<" GeV"<<G4endl;

  G4double e_kin_rescaled = targetNucleus.EvaporationEffects(originalIncident->GetTotalEnergy()-originalIncident->GetDefinition()->GetPDGMass());

  G4cout<<"Rescaled kinetic energy: "<<e_kin_rescaled<<G4endl;

  const G4double cutOff = 0.1*MeV;

  if ( e_kin_rescaled < cutOff )
    {
      aParticleChange.ProposeTrackStatus( fStopAndKill );//If the dice decides not to cascade I stop the particle
      return &aParticleChange;
    }
  */
  // create the target particle
  
  G4DynamicParticle *originalTarget = new G4DynamicParticle;
  originalTarget->SetDefinition(aTarget);
  
  G4ReactionProduct targetParticle(aTarget);
  
  
  G4ReactionProduct currentParticle(const_cast<G4ParticleDefinition *>(originalIncident->GetDefinition() ) );
  currentParticle.SetMomentum( originalIncident->Get4Momentum().vect() );
  currentParticle.SetKineticEnergy( originalIncident->GetKineticEnergy() );

  /*
  G4cout<<"After creation:"<<G4endl;
  G4cout<<"currentParticle: "<<currentParticle.GetMomentum()/GeV<<" GeV vs. "<<OrgPart->Get4Momentum()/GeV<<" GeV"<<G4endl;
  G4cout<<"targetParticle: "<<targetParticle.GetMomentum()/GeV<<" GeV"<<G4endl;
  G4cout<<"Fourmomentum from originalIncident: "<<originalIncident->Get4Momentum()<<" vs "<<OrgPart->Get4Momentum()<<G4endl;
  */


  G4ReactionProduct modifiedOriginal = currentParticle;
  
  currentParticle.SetSide( 1 ); // incident always goes in forward hemisphere
  targetParticle.SetSide( -1 );  // target always goes in backward hemisphere
  G4bool incidentHasChanged = false;
  if (!IncidentSurvives) incidentHasChanged = true; //I wonder if I am supposed to do this...
  G4bool targetHasChanged = false;
  if (!TargetSurvives) targetHasChanged = true; //Ditto here
  G4bool quasiElastic = false;
  if (rp.size()==2) quasiElastic = true; //Oh well...
  G4FastVector<G4ReactionProduct,MYGHADLISTSIZE> vec;  // vec will contain the secondary particles
  G4int vecLen = 0;
  vec.Initialize( 0 );

  
  
  // I hope my understanding of "secondary" is correct here
  // I think that it entails only what is not a surviving incident of target
  
  for (G4int i = 0; i!=NumberOfSecondaries;i++){
    if(theParticleDefinitions[i]!=aTarget 
       && theParticleDefinitions[i]!=originalIncident->GetDefinition()
       && theParticleDefinitions[i]!=outgoingRhadron
       && theParticleDefinitions[i]!=outgoingTarget)
      { 
	G4ReactionProduct* pa = new G4ReactionProduct;
	pa->SetDefinition( theParticleDefinitions[i] );
	(G4UniformRand() < 0.5) ? pa->SetSide( -1 ) : pa->SetSide( 1 );
	vec.SetElement( vecLen++, pa );
      }
  }

  //  if (incidentHasChanged) currentParticle.SetDefinitionAndUpdateE( outgoingRhadron );

  if (incidentHasChanged) currentParticle.SetDefinitionAndUpdateE( outgoingCloud );
  if (incidentHasChanged) modifiedOriginal.SetDefinition( outgoingCloud );//Is this correct? It solves the "free energy" checking in ReactionDynamics
  if (targetHasChanged) targetParticle.SetDefinitionAndUpdateE( outgoingTarget );

  //  G4cout<<"Calling CalculateMomenta... "<<G4endl;
  /*
  G4cout<<"Current particle starts as: "<<currentParticle.GetDefinition()->GetParticleName()<<G4endl;
  G4cout<<"with momentum: "<<currentParticle.GetMomentum()/GeV<<" GeV"<<G4endl;
  G4cout<<"May be killed?: "<<currentParticle.GetMayBeKilled()<<G4endl;
  */
  //  G4double e_temp = currentParticle.GetKineticEnergy();

  CalculateMomenta( vec, vecLen,
		    originalIncident, originalTarget, modifiedOriginal,
		    targetNucleus, currentParticle, targetParticle,
		    incidentHasChanged, targetHasChanged, quasiElastic );

  //  G4cout <<"Cloud loss: "<<(e_temp-currentParticle.GetKineticEnergy())/GeV<<" GeV"<<G4endl;

  G4String cPname = currentParticle.GetDefinition()->GetParticleName();

  //  if(cPname!="rhadronmesoncloud"&&cPname!="rhadronbaryoncloud")
  //    {
  /*
  G4cout<<"Current particle is now: "<<cPname <<G4endl;
  G4cout<<"with momentum: "<<currentParticle.GetMomentum()/GeV<<" GeV"<<G4endl;
  G4cout<<"and kinetic energy: "<<currentParticle.GetKineticEnergy()/GeV<<" GeV"<<G4endl;
  G4cout<<"May be killed?: "<<currentParticle.GetMayBeKilled()<<G4endl;
  G4cout<<"Modified original is: "<<modifiedOriginal.GetDefinition()->GetParticleName()<<G4endl;
  G4cout<<"with momentum: "<<modifiedOriginal.GetMomentum()/GeV<<" GeV"<<G4endl;
  G4cout<<"and kinetic energy: "<<modifiedOriginal.GetKineticEnergy()/GeV<<" GeV"<<G4endl;
  G4cout<<"May be killed?: "<<modifiedOriginal.GetMayBeKilled()<<G4endl;
  G4cout<<"Target particle is: "<<targetParticle.GetDefinition()->GetParticleName()<<G4endl;
  G4cout<<"with momentum: "<<targetParticle.GetMomentum()/GeV<<" GeV"<<G4endl;
  G4cout<<"and kinetic energy: "<<targetParticle.GetKineticEnergy()/GeV<<" GeV"<<G4endl;
  G4cout<<"May be killed?: "<<targetParticle.GetMayBeKilled()<<G4endl;
  G4cout<<"incidentHasChanged: "<<incidentHasChanged<<G4endl;
  G4cout<<"targetHasChanged: "<<targetHasChanged<<G4endl;
  G4cout<<"Particles in vec:"<<G4endl;
  for(int i=0; i<vecLen; ++i )
    {
      G4cout<< vec[i]->GetDefinition()->GetParticleName()<<G4endl;
    }
  */

      //    }

  //  G4cout<<"Done!"<<G4endl;
  //    const G4LorentzRotation& trans(originalIncident->GetTrafoToLab());
  //    G4cout<<"Check aParticleChange.GetNumberOfSecondaries(): "<<aParticleChange.GetNumberOfSecondaries()<<G4endl;

  aParticleChange.SetNumberOfSecondaries(vecLen+NumberOfSecondaries);
  G4double e_kin=0;
  G4LorentzVector cloud_p4_new; //Cloud 4-momentum in lab after collision
  //  n++;
  //  G4cout << n << G4endl;
  /*
  if(cPname!="rhadronmesoncloud"&&cPname!="rhadronbaryoncloud") {
    G4cout<<"Cloud deleted!!! AAARRRRGGGHHH!!!"<<G4endl;
    G4cout<<"Cloud name: "<<cPname<<G4endl;
    G4cout<<"E_kin_0: "<<e_kin_0/GeV<<" GeV"<<G4endl;
    //    G4cout<<"n: "<<n<<G4endl;
    //    n=0;
  }
  */
  cloud_p4_new.setVectM(currentParticle.GetMomentum(),currentParticle.GetMass());
  cloud_p4_new *= trans;

  G4LorentzVector cloud_p4_old_full = Cloud4Momentum; //quark system in CMS BEFORE collision
  cloud_p4_old_full.boost(trafo_full_cms);
  G4LorentzVector cloud_p4_old_cloud = Cloud4Momentum; //quark system in cloud CMS BEFORE collision
  cloud_p4_old_cloud.boost(trafo_cloud_cms);
  G4LorentzVector cloud_p4_full = cloud_p4_new; //quark system in CMS AFTER collision
  cloud_p4_full.boost(trafo_full_cms);
  G4LorentzVector cloud_p4_cloud = cloud_p4_new; //quark system in cloud CMS AFTER collision
  cloud_p4_cloud.boost(trafo_cloud_cms);

  G4LorentzVector p_g_cms = gluinoMomentum; //gluino in CMS BEFORE collision
  p_g_cms.boost(trafo_full_cms);

  G4LorentzVector p4_new;
  //    p4_new.setVectM(cloud_p4_full.v()+p_g_cms.v(),outgoingRhadron->GetPDGMass());
  //    p4_new.boost(-trafo_full_cms);
    //    p4_new = cloud_p4_new + gluinoMomentum;
  p4_new.setVectM( cloud_p4_new.v() + gluinoMomentum.v(), outgoingRhadron->GetPDGMass() );
  //  G4cout<<"P4-diff: "<<(p4_new-cloud_p4_new-gluinoMomentum)/GeV<<", magnitude: "<<(p4_new-cloud_p4_new-gluinoMomentum).m()/MeV<<" MeV" <<G4endl;

  G4ThreeVector p_new;
  p_new = p4_new.vect();

  aParticleChange.ProposeLocalEnergyDeposit((p4_new-cloud_p4_new-gluinoMomentum).m());

  if( incidentHasChanged )
    {
      G4DynamicParticle* p0 = new G4DynamicParticle;
      p0->SetDefinition( outgoingRhadron );
      p0->SetMomentum( p_new ); 

      // May need to run SetDefinitionAndUpdateE here...
      G4Track* Track0 = new G4Track(p0,
				    aTrack.GetGlobalTime(),
				    aPosition);
      Track0->SetTouchableHandle(thisTouchable);
      aParticleChange.AddSecondary(Track0);
      /*
      G4cout<<"Adding a particle "<<p0->GetDefinition()->GetParticleName()<<G4endl;
      G4cout<<"with momentum: "<<p0->GetMomentum()/GeV<<" GeV"<<G4endl;
      G4cout<<"and kinetic energy: "<<p0->GetKineticEnergy()/GeV<<" GeV"<<G4endl;
      */
      if(p0->GetKineticEnergy()>e_kin_0) {
	G4cout<<"ALAAAAARM!!! (incident changed from "
	      <<IncidentRhadron->GetDefinition()->GetParticleName()
	      <<" to "<<p0->GetDefinition()->GetParticleName()<<")"<<G4endl;
	G4cout<<"Energy loss: "<<(e_kin_0-p0->GetKineticEnergy())/GeV<<" GeV (should be positive)"<<G4endl;
	//Turns out problem is only in 2 -> 3 (Won't fix 2 -> 2 energy deposition)
	if(rp.size()!=3) G4cout<<"DOUBLE ALAAAAARM!!!"<<G4endl;
      } /*else {
	G4cout<<"NO ALAAAAARM!!!"<<G4endl;
	}*/
      if(std::abs(p0->GetKineticEnergy()-e_kin_0)>100*GeV) {
	G4cout<<"Diff. too big"<<G4endl;
      }

      aParticleChange.ProposeTrackStatus( fStopAndKill );
    }
  else
    {

      G4double p = p_new.mag();
      if( p > DBL_MIN )
	aParticleChange.ProposeMomentumDirection( p_new.x()/p, p_new.y()/p, p_new.z()/p );
      else
	aParticleChange.ProposeMomentumDirection( 1.0, 0.0, 0.0 );
      
      G4double aE = sqrt(p*p+(outgoingRhadron->GetPDGMass()*outgoingRhadron->GetPDGMass()) );
      e_kin = aE - outgoingRhadron->GetPDGMass();
      /* AR_NEWCODE_IMPORT 
      if(e_kin>e_kin_0) {
	G4cout<<"ALAAAAARM!!!"<<G4endl;
	G4cout<<"Energy loss: "<<(e_kin_0-e_kin)/GeV<<" GeV (should be positive)"<<G4endl;
	if(rp.size()!=3) G4cout<<"DOUBLE ALAAAAARM!!!"<<G4endl;
      }
      if(std::abs(e_kin-e_kin_0)>100*GeV) {
	G4cout<<"Diff. too big"<<G4endl;
      }

      if (std::fabs(aE)<.1*eV) aE=.1*eV;
      aParticleChange.ProposeEnergy( aE- outgoingRhadron->GetPDGMass() ); //I do not know if this is correct...
      if(std::abs(e_kin-e_kin_0)>100*GeV) {
	G4cout<<"Diff. too big"<<G4endl;
      }
          */
    }
  
  //    return G4VDiscreteProcess::PostStepDoIt( aTrack, aStep);
  if( targetParticle.GetMass() > 0.0 )  // targetParticle can be eliminated in TwoBody
    {
      G4DynamicParticle *p1 = new G4DynamicParticle;
      p1->SetDefinition( targetParticle.GetDefinition() );
      //      G4cout<<"Target secondary: "<<targetParticle.GetDefinition()->GetParticleName()<<G4endl;
      G4ThreeVector momentum = targetParticle.GetMomentum();
      momentum = momentum.rotate(cache,what);
      p1->SetMomentum( momentum );
      p1->SetMomentum( (trans*p1->Get4Momentum()).vect());
      G4Track* Track1 = new G4Track(p1,
				    aTrack.GetGlobalTime(),
				    aPosition);
      Track1->SetTouchableHandle(thisTouchable); 
      aParticleChange.AddSecondary(Track1);
    }
  G4DynamicParticle *pa;
  /*
    G4cout<<"vecLen: "<<vecLen<<G4endl;
    G4cout<<"#sec's: "<<aParticleChange.GetNumberOfSecondaries()<<G4endl;
  */
  
  
  
  for(int i=0; i<vecLen; ++i )
    {
      pa = new G4DynamicParticle();
      pa->SetDefinition( vec[i]->GetDefinition() );
      pa->SetMomentum( vec[i]->GetMomentum() );
      pa->Set4Momentum(trans*(pa->Get4Momentum()));
      G4ThreeVector pvec = pa->GetMomentum();
      G4Track* Trackn = new G4Track(pa,
				    aTrack.GetGlobalTime(),
				    aPosition);
      Trackn->SetTouchableHandle(thisTouchable);
      aParticleChange.AddSecondary(Trackn);

      // debug 

//       G4cerr << "FullModelHadronicProcess: New secondary " << i 
//              << " ID " << Trackn->GetTrackID() 
//              << " PDG " << Trackn->GetDefinition()->GetParticleName() 
//              << " position " << Trackn->GetPosition() 
//              << " volume " << Trackn->GetTouchable() 
//              << " handle " << Trackn->GetTouchableHandle() << G4endl;

      delete vec[i];
    } 

  // Histogram filling  
  const G4DynamicParticle* theRhadron = FindRhadron(&aParticleChange);
  

  if (theRhadron!=NULL||IncidentSurvives)
    {
      
      double E_new;
      if(IncidentSurvives)
	{
	  //	  E_new = currentParticle.GetKineticEnergy();
	  E_new = e_kin;
	} else {
	  E_new = theRhadron->GetKineticEnergy();
	  if(CustomPDGParser::s_isRMeson(theRhadron->GetDefinition()->GetPDGEncoding())
	     !=CustomPDGParser::s_isRMeson(theIncidentPDG)
	     ||
	     CustomPDGParser::s_isMesonino(theRhadron->GetDefinition()->GetPDGEncoding())
	     !=CustomPDGParser::s_isMesonino(theIncidentPDG)
	     ) {

	    G4cout<<"Rm: "<<CustomPDGParser::s_isRMeson(theRhadron->GetDefinition()->GetPDGEncoding())
		     <<" vs: "<<CustomPDGParser::s_isRMeson(theIncidentPDG)<<G4endl;
	    G4cout<<"Sm: "<<CustomPDGParser::s_isMesonino(theRhadron->GetDefinition()->GetPDGEncoding())
		     <<" vs: "<<CustomPDGParser::s_isMesonino(theIncidentPDG)<<G4endl;

	  }
	}
      
      //Calculating relevant scattering angles.
      G4LorentzVector p4_old_full = FullRhadron4Momentum; //R-hadron in CMS BEFORE collision
      p4_old_full.boost(trafo_full_cms);
      G4LorentzVector p4_old_cloud = FullRhadron4Momentum; //R-hadron in cloud CMS BEFORE collision
      p4_old_cloud.boost(trafo_cloud_cms);
      G4LorentzVector p4_full = p4_new; //R-hadron in CMS AFTER collision
      //      G4cout<<p4_full.v()/GeV<<G4endl;
      p4_full=p4_full.boost(trafo_full_cms);
      //      G4cout<<p4_full.m()<<" / "<<(cloud_p4_new+gluinoMomentum).boost(trafo_full_cms).m()<<G4endl;
      G4LorentzVector p4_cloud = p4_new; //R-hadron in cloud CMS AFTER collision
      p4_cloud.boost(trafo_cloud_cms);

      
      /*
      G4double dtheta_fullcms = p4_full.vect().angle(p4_old_full.vect());
      G4double dtheta_cloudcms = p4_cloud.vect().angle(p4_old_cloud.vect());
      G4double dtheta_lab = p_new.angle(p_0);//acos(p_0*p_new/(p_0.mag()*p_new.mag())); 

      G4double cloud_dtheta_fullcms = cloud_p4_full.vect().angle(cloud_p4_old_full.vect());
      G4double cloud_dtheta_cloudcms = cloud_p4_cloud.vect().angle(p4_old_cloud.vect());
      G4double cloud_dtheta_lab = cloud_p4_new.vect().angle(p_0);
      //Writing out momenta for manual check of boosts:
      G4cout<<"******************************************"<<G4endl;

      G4cout<<"R-hadron, before: "<<G4endl;
      G4cout<<"Lab: "<<FullRhadron4Momentum.v().mag()/GeV<<" GeV"<<G4endl;
      G4cout<<"CMS: "<<p4_old_full.v().mag()/GeV<<" GeV"<<G4endl;
      G4cout<<"after: "<<G4endl;
      G4cout<<"Lab: "<<p4_new.v().mag()/GeV<<" GeV"<<G4endl;
      G4cout<<"CMS: "<<p4_full.v().mag()/GeV<<" GeV"<<G4endl;
      G4cout<<"cos(theta*): "<<cos(dtheta_fullcms)<<G4endl;
      G4cout<<"Gluino: "<<G4endl;
      G4cout<<"Lab: "<<gluinoMomentum.v().mag()/GeV<<" GeV"<<G4endl;
      G4cout<<"CMS: "<<p_g_cms.v().mag()/GeV<<" GeV"<<G4endl;

      G4cout<<"Cloud, before: "<<G4endl;
      G4cout<<"Lab: "<<Cloud4Momentum.v().mag()/GeV<<" GeV"<<G4endl;
      G4cout<<"CMS: "<<cloud_p4_old_full.v().mag()/GeV<<" GeV"<<G4endl;
      G4cout<<"after: "<<G4endl;
      G4cout<<"CMS: "<<cloud_p4_full.v().mag()/GeV<<" GeV"<<G4endl;
      G4cout<<"cloud cos(theta*): "<<cos(cloud_dtheta_fullcms)<<G4endl;
      G4cout<<"Longitudinal: "<<cos(cloud_dtheta_fullcms)*cloud_p4_full.v().mag()/GeV<<" GeV"<<G4endl;
      G4cout<<"~Combined longitudinal: "<<(cos(cloud_dtheta_fullcms)*cloud_p4_full.v().mag()+p4_old_full.v().mag())/GeV<<" GeV"<<G4endl;
      if ((cos(cloud_dtheta_fullcms)*cloud_p4_full.v().mag()+p4_old_full.v().mag())<0.) G4cout<<"ALAAARM"<<G4endl;
      G4cout<<"Psum cos(theta*): "<<cos((p_g_cms.v()+cloud_p4_full.v()).angle(p4_old_full.v()))<<G4endl;
      G4cout<<"True R-hadron (CMS): "<<(p_g_cms.v()+cloud_p4_full.v()).mag()/GeV<<" GeV"<<G4endl;
      G4cout<<"******************************************"<<G4endl;
      G4cout<<"Fucking manual fucking calculation:"<<G4endl;
      G4cout<<"Momenta:"<<G4endl;
      G4cout<<"Cloud, lab: "<<cloud_p4_new.v()/GeV<<" + gluino: "<<gluinoMomentum.v()/GeV
	    <<" = "<<(cloud_p4_new.v()+gluinoMomentum.v())/GeV
	    <<". Boost to CMS: "<<(cloud_p4_new+gluinoMomentum).boost(trafo_full_cms).v()/GeV<<G4endl;
      G4cout<<"Cloud, CMS: "<<cloud_p4_full.v()/GeV<<" + gluino: "<<p_g_cms.v()/GeV
	    <<" = "<<(cloud_p4_full.v()+p_g_cms.v())/GeV
	    <<". Boost to Lab: "<<(cloud_p4_full+p_g_cms).boost(-trafo_full_cms).v()/GeV<<G4endl;
      G4cout<<"Ref: "<<p4_new.v()/GeV<<" / "<<p4_full.v()/GeV<<G4endl;
      G4cout<<"******************************************"<<G4endl;
      */


      //      std::cout<<"Lab, fullcms, cloudcms: "<<dtheta_lab<<", "<<dtheta_fullcms<<", "<<dtheta_cloudcms<<std::endl;
      G4double AbsDeltaE = E_0-E_new;
      //      G4cout <<"Energy loss: "<<AbsDeltaE/GeV<<G4endl;
      if (AbsDeltaE > 10*GeV) {
	G4cout<<"Energy loss larger than 10 GeV..."<<G4endl;
	G4cout<<"E_0: "<<E_0/GeV<<" GeV"<<G4endl;
	G4cout<<"E_new: "<<E_new/GeV<<" GeV"<<G4endl;
	G4cout<<"Gamma: "<<IncidentRhadron->GetTotalEnergy()/IncidentRhadron->GetDefinition()->GetPDGMass()<<G4endl;
	G4cout<<"x: "<<aPosition.x()/cm<<" cm"<<G4endl;
      }
     } 
  delete originalIncident;
  delete originalTarget;
  //  aParticleChange.DumpInfo();
  //  G4cout << "Exiting FullModelHadronicProcess::PostStepDoIt"<<G4endl;

  //clear interaction length      
  ClearNumberOfInteractionLengthLeft();

  
  return &aParticleChange;
  
}


void FullModelHadronicProcess::CalculateMomenta(
					       G4FastVector<G4ReactionProduct,MYGHADLISTSIZE> &vec,
					       G4int &vecLen,
					       const G4HadProjectile *originalIncident,   // the original incident particle
					       const G4DynamicParticle *originalTarget,
					       G4ReactionProduct &modifiedOriginal,   // Fermi motion and evap. effects included
					       G4Nucleus &targetNucleus,
					       G4ReactionProduct &currentParticle,
					       G4ReactionProduct &targetParticle,
					       G4bool &incidentHasChanged,
					       G4bool &targetHasChanged,
					       G4bool quasiElastic )
{
  FullModelReactionDynamics theReactionDynamics;

  cache = 0;
  what = originalIncident->Get4Momentum().vect();

  //Commented out like in casqmesmin.F where CALL STPAIR is commented out
  /*
  theReactionDynamics.ProduceStrangeParticlePairs( vec, vecLen,
						   modifiedOriginal, originalTarget,
						   currentParticle, targetParticle,
						   incidentHasChanged, targetHasChanged );
  */

  if( quasiElastic )
    {
      //      G4cout<<"We are calling TwoBody..."<<G4endl;
      theReactionDynamics.TwoBody( vec, vecLen,
                                   modifiedOriginal, originalTarget,
                                   currentParticle, targetParticle,
                                   targetNucleus, targetHasChanged );

      return;
    }

  //If ProduceStrangeParticlePairs is commented out, let's cut this one as well
  G4ReactionProduct leadingStrangeParticle;
  G4bool leadFlag = MarkLeadingStrangeParticle( currentParticle,
						targetParticle,
						leadingStrangeParticle );



  //
  // Note: the number of secondaries can be reduced in GenerateXandPt and TwoCluster
  //
  G4bool finishedGenXPt = false;
  G4bool annihilation = false;
  if( originalIncident->GetDefinition()->GetPDGEncoding() < 0 &&
      currentParticle.GetMass() == 0.0 && targetParticle.GetMass() == 0.0 )
    {
      // original was an anti-particle and annihilation has taken place
      annihilation = true;
      G4double ekcor = 1.0;
      G4double ek = originalIncident->GetKineticEnergy();
      G4double ekOrg = ek;

      const G4double tarmas = originalTarget->GetDefinition()->GetPDGMass();
      if( ek > 1.0*GeV )ekcor = 1./(ek/GeV);
      const G4double atomicWeight = G4double(targetNucleus.GetN_asInt());
      ek = 2*tarmas + ek*(1.+ekcor/atomicWeight);

      G4double tkin = targetNucleus.Cinema( ek );
      ek += tkin;
      ekOrg += tkin;
      modifiedOriginal.SetKineticEnergy( ekOrg );
    }

  const G4double twsup[] = { 1.0, 0.7, 0.5, 0.3, 0.2, 0.1 };
  G4double rand1 = G4UniformRand();
  G4double rand2 = G4UniformRand();
  if( (annihilation || (vecLen >= 6) || (modifiedOriginal.GetKineticEnergy()/GeV >= 1.0)) && 
      (((originalIncident->GetDefinition() == G4KaonPlus::KaonPlus()) || 
         (originalIncident->GetDefinition() == G4KaonMinus::KaonMinus()) || 
         (originalIncident->GetDefinition() == G4KaonZeroLong::KaonZeroLong()) || 
         (originalIncident->GetDefinition() == G4KaonZeroShort::KaonZeroShort())) && 
       ((rand1 < 0.5) || (rand2 > twsup[vecLen]))))
    finishedGenXPt =
      theReactionDynamics.GenerateXandPt( vec, vecLen,
					  modifiedOriginal, originalIncident,
					  currentParticle, targetParticle,
					  targetNucleus, incidentHasChanged,
					  targetHasChanged, leadFlag,
					  leadingStrangeParticle );
  if( finishedGenXPt )
    {
      Rotate(vec, vecLen);
      return;
    }

  G4bool finishedTwoClu = false;
  if( modifiedOriginal.GetTotalMomentum()/MeV < 1.0 )
    {

      for(G4int i=0; i<vecLen; i++) delete vec[i];
      vecLen = 0;
    }
  else
    {

      theReactionDynamics.SuppressChargedPions( vec, vecLen,
						modifiedOriginal, currentParticle,
                                                targetParticle, targetNucleus,
                                                incidentHasChanged, targetHasChanged );

      try
	{
	  finishedTwoClu = theReactionDynamics.TwoCluster( vec, vecLen,
							   modifiedOriginal, originalIncident,
							   currentParticle, targetParticle,
							   targetNucleus, incidentHasChanged,
							   targetHasChanged, leadFlag,
							   leadingStrangeParticle );
	}
      catch(G4HadReentrentException aC)
	{
	  aC.Report(G4cout);
	  throw G4HadReentrentException(__FILE__, __LINE__, "Failing to calculate momenta");
	}
    }
  if( finishedTwoClu )
    {
      Rotate(vec, vecLen);
      return;
    }

  //
  // PNBlackTrackEnergy is the kinetic energy available for
  //   proton/neutron black track particles [was enp(1) in fortran code]
  // DTABlackTrackEnergy is the kinetic energy available for
  //   deuteron/triton/alpha particles      [was enp(3) in fortran code]
  //const G4double pnCutOff = 0.1;
  //const G4double dtaCutOff = 0.1;
  //if( (targetNucleus.GetN() >= 1.5)
  //    && !(incidentHasChanged || targetHasChanged)
  //    && (targetNucleus.GetPNBlackTrackEnergy()/MeV <= pnCutOff)
  //    && (targetNucleus.GetDTABlackTrackEnergy()/MeV <= dtaCutOff) )
  //{
  // the atomic weight of the target nucleus is >= 1.5            AND
  //   neither the incident nor the target particles have changed  AND
  //     there is no kinetic energy available for either proton/neutron
  //     or for deuteron/triton/alpha black track particles
  // For diffraction scattering on heavy nuclei use elastic routines instead
  //G4cerr << "*** Error in G4InelasticInteraction::CalculateMomenta" << G4endl;
  //G4cerr << "*** the elastic scattering would be better here ***" <<G4endl;
  //}
  theReactionDynamics.TwoBody( vec, vecLen,
			       modifiedOriginal, originalTarget,
			       currentParticle, targetParticle,
			       targetNucleus, targetHasChanged );
}


G4bool FullModelHadronicProcess::MarkLeadingStrangeParticle(
							    const G4ReactionProduct &currentParticle,
							    const G4ReactionProduct &targetParticle,
							    G4ReactionProduct &leadParticle )
{
  // the following was in GenerateXandPt and TwoCluster
  // add a parameter to the GenerateXandPt function telling it about the strange particle
  //
  // assumes that the original particle was a strange particle
  //
  G4bool lead = false;
  if( (currentParticle.GetMass() >= G4KaonPlus::KaonPlus()->GetPDGMass()) &&
      (currentParticle.GetDefinition() != G4Proton::Proton()) &&
      (currentParticle.GetDefinition() != G4Neutron::Neutron()) )
    {
      lead = true;
      leadParticle = currentParticle;              //  set lead to the incident particle
    }
  else if( (targetParticle.GetMass() >= G4KaonPlus::KaonPlus()->GetPDGMass()) &&
	   (targetParticle.GetDefinition() != G4Proton::Proton()) &&
	   (targetParticle.GetDefinition() != G4Neutron::Neutron()) )
    {
      lead = true;
      leadParticle = targetParticle;              //   set lead to the target particle
    }
  return lead;
}

void FullModelHadronicProcess::Rotate(G4FastVector<G4ReactionProduct,MYGHADLISTSIZE> &vec, G4int &vecLen)
{
  G4double rotation = 2.*pi*G4UniformRand();
  cache = rotation;
  G4int i;
  for( i=0; i<vecLen; ++i )
    {
      G4ThreeVector momentum = vec[i]->GetMomentum();
      momentum = momentum.rotate(rotation, what);
      vec[i]->SetMomentum(momentum);
    }
}      

const G4DynamicParticle* FullModelHadronicProcess::FindRhadron(G4ParticleChange* aParticleChange)
{
  G4int nsec = aParticleChange->GetNumberOfSecondaries();
  if (nsec==0) return 0;
  int i = 0;
  G4bool found = false;
  while (i!=nsec && !found){
    //    G4cout<<"Checking "<<aParticleChange->GetSecondary(i)->GetDynamicParticle()->GetDefinition()->GetParticleName()<<G4endl;
    //    if (aParticleChange->GetSecondary(i)->GetDynamicParticle()->GetDefinition()->GetParticleType()=="rhadron") found = true;
    if (dynamic_cast<CustomParticle*>(aParticleChange->GetSecondary(i)->GetDynamicParticle()->GetDefinition())!=0) found = true;
    i++;
  }
  i--;
  if(found) return aParticleChange->GetSecondary(i)->GetDynamicParticle();
  return 0;
}

