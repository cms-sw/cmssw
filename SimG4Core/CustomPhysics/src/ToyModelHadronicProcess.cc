//Geant4 include
#include "G4ProcessManager.hh"
#include "G4ParticleTable.hh"
#include "G4Track.hh"
#include "Randomize.hh"
//Our includes
#include "SimG4Core/CustomPhysics/interface/ToyModelHadronicProcess.hh"
#include "SimG4Core/CustomPhysics/interface/CustomPDGParser.h"
#include "SimG4Core/CustomPhysics/interface/CustomParticle.h"
#include "SimG4Core/CustomPhysics/interface/HadronicProcessHelper.hh"
#include "SimG4Core/CustomPhysics/interface/Decay3Body.h"

using namespace CLHEP;

ToyModelHadronicProcess::ToyModelHadronicProcess(HadronicProcessHelper * aHelper, const G4String& processName) :
  G4VDiscreteProcess(processName), m_verboseLevel(0),  m_helper(aHelper),  m_detachCloud(true)
{
  // Instantiating helper class
  //m_helper = HadronicProcessHelper::instance();
  //m_verboseLevel=0;
  //m_detachCloud=true;

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

  // Need to provide Set-methods for these in time
  G4double highestEnergyLimit = 10 * TeV  ;
  G4double lowestEnergyLimit = 1 * eV;
  G4double particleEnergy = particle->GetKineticEnergy();
   
  if (particleEnergy > highestEnergyLimit || particleEnergy < lowestEnergyLimit){
    if(m_verboseLevel >= 1) std::cout << "ToyModelHadronicProcess: Energy out of bounds [" << 
			      lowestEnergyLimit / MeV << "MeV , " << 
			      highestEnergyLimit / MeV << "MeV ] while it is " << particleEnergy/MeV  << 
			      std::endl;
    return 0;
  } else {
    if(m_verboseLevel >= 3) std::cout << "ToyModelHadronicProcess: Return cross section " << inclusiveCrossSection << std::endl;
    return inclusiveCrossSection;
  }

}


G4double ToyModelHadronicProcess::GetMeanFreePath(const G4Track& aTrack, G4double, G4ForceCondition*)
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


G4VParticleChange* ToyModelHadronicProcess::PostStepDoIt(const G4Track& track,
							const G4Step& /*  step*/)
{

  const G4TouchableHandle thisTouchable(track.GetTouchableHandle());
  
  // A little setting up
  G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
  m_particleChange.Initialize(track);
  //  G4DynamicParticle* incidentRHadron = const_cast<G4DynamicParticle*>(track.GetDynamicParticle()); //This will contain RHadron Def + RHad momentum
  const G4DynamicParticle* incidentRHadron = track.GetDynamicParticle(); //This will contain RHadron Def + RHad momentum
//  double E_0 = incidentRHadron->GetKineticEnergy();
//  const G4int theIncidentPDG = incidentRHadron->GetDefinition()->GetPDGEncoding();
  const G4ThreeVector aPosition = track.GetPosition();

//  double gamma = incidentRHadron->GetTotalEnergy()/incidentRHadron->GetDefinition()->GetPDGMass();

  CustomParticle* CustomIncident = dynamic_cast<CustomParticle*>(incidentRHadron->GetDefinition());
  G4DynamicParticle* cloudParticle =  new G4DynamicParticle(); //This will contain Cloud Def + scaled momentum
           
  //  G4int rHadronPdg = incidentRHadron->GetDefinition()->GetPDGEncoding();

  //set cloud definition
  if(CustomIncident==0)
    {
      std::cout << "ToyModelHadronicProcess::PostStepDoIt  Definition of particle cloud not available!!" << std::endl;
    } else {
      cloudParticle->SetDefinition(CustomIncident->GetCloud());
    }
     
  //compute scaled momentum  
  double scale=cloudParticle->GetDefinition()->GetPDGMass()/incidentRHadron->GetDefinition()->GetPDGMass();
  G4LorentzVector cloudMomentum;
  cloudMomentum.setVectM(incidentRHadron->GetMomentum()*scale,cloudParticle->GetDefinition()->GetPDGMass());
  G4LorentzVector gluinoMomentum;
  gluinoMomentum.setVectM(incidentRHadron->GetMomentum()*(1.-scale),CustomIncident->GetSpectator()->GetPDGMass());
  
  cloudParticle->Set4Momentum(cloudMomentum);         	
  
  const G4DynamicParticle *incidentParticle;
  if(! m_detachCloud) incidentParticle = incidentRHadron; 
  else incidentParticle= cloudParticle;
  
  if(m_verboseLevel >= 3)
    { 
      std::cout << "ToyModelHadronicProcess::PostStepDoIt    After scaling " << std::endl;
      std::cout << "      RHadron "<<incidentRHadron->GetDefinition()->GetParticleName()<<"         pdgmass= " << incidentRHadron->GetDefinition()->GetPDGMass() / GeV
		<< " P= " << incidentRHadron->Get4Momentum() / GeV<<" mass= "<< incidentRHadron->Get4Momentum().m() / GeV <<std::endl;
      std::cout << "      Cloud   "<<cloudParticle->GetDefinition()->GetParticleName()<<"         pdgmass= " << cloudParticle->GetDefinition()->GetPDGMass() / GeV
		<< " P= " << cloudParticle->Get4Momentum() / GeV<<" mass= "<< cloudParticle->Get4Momentum().m() / GeV <<std::endl;
      std::cout << "      Incident          pdgmass= " << incidentParticle->GetDefinition()->GetPDGMass() / GeV
		<< " P= " << incidentParticle->Get4Momentum() / GeV<<" mass= "<< incidentParticle->Get4Momentum().m() / GeV <<std::endl;
    }
 
  const G4ThreeVector position = track.GetPosition();
  const G4int incidentParticlePDG = incidentRHadron->GetDefinition()->GetPDGEncoding();
  std::vector<G4ParticleDefinition*> newParticles;        // this will contain clouds
  std::vector<G4ParticleDefinition*> newParticlesRHadron; // this will contain r-hadron

  G4bool incidentSurvives = false;
  
  //Get the final state particles and target
  G4ParticleDefinition* targetParticle; 
  HadronicProcessHelper::ReactionProduct reactionProduct = m_helper->finalState(incidentRHadron,track.GetMaterial(),targetParticle);

  // Fill a list of the new particles to create 
  //(i.e. reaction products without the incident if it survives)
   
  for(HadronicProcessHelper::ReactionProduct::iterator it  = reactionProduct.begin();
      it != reactionProduct.end() ;  
      it++)
    {
      G4ParticleDefinition * productDefinition =theParticleTable->FindParticle(*it);

      if (productDefinition->GetPDGEncoding()==incidentParticlePDG)
        incidentSurvives = true;

      newParticlesRHadron.push_back(productDefinition);
      
      CustomParticle* cProd = 0;
      cProd = dynamic_cast<CustomParticle*>(productDefinition);

      if( cProd!=0 && m_detachCloud)
	productDefinition=cProd->GetCloud();

      newParticles.push_back(productDefinition);
    }
  
  int numberOfSecondaries = reactionProduct.size();
  
  if(m_verboseLevel >= 2) std::cout << "ToyModelHadronicProcess::PostStepDoIt  N secondaries: " 
				    << numberOfSecondaries << std::endl;
  

  //************ My toy model ********************
  // 2 -> 2 goes to CM, chooses a random direction and fires the particles off back to back
  // 2 -> 3 Effectively two two-body decays

  // Getting fourmomenta
  const G4LorentzVector incident4Momentum = incidentParticle->Get4Momentum();
  const G4LorentzVector target4Momentum(0,0,0,targetParticle->GetPDGMass());
  const G4LorentzVector sum4Momentum = incident4Momentum + target4Momentum;
  const G4ThreeVector   cmBoost = sum4Momentum.boostVector();//The boost from CM to lab
  const G4LorentzVector cm4Momentum = sum4Momentum.rest4Vector();

  if(m_verboseLevel >= 2) std::cout << "ToyModelHadronicProcess::PostStepDoIt  Kinematics in GeV: " << std::endl 
				    << "     Boost   = " << cmBoost / GeV<< std::endl
				    << "     4P CM   = " << cm4Momentum / GeV << std::endl
				    << "     4P Inc  = " << incident4Momentum / GeV << std::endl
				    << "     4P Targ = " << target4Momentum  / GeV<< std::endl;

  //Choosing random direction
  const G4double phi_p = 2*pi*G4UniformRand()-pi ;
  const G4double theta_p = pi*G4UniformRand() ;
  const G4ThreeVector randomDirection(sin(theta_p)*cos(phi_p),
				      sin(theta_p)*sin(phi_p),
				      cos(theta_p));
  

  std::vector<G4double> m;
  std::vector<G4LorentzVector> fourMomenta;
  
  //Fill the masses
  for(int ip=0;ip<numberOfSecondaries;ip++)     
    {
      m.push_back(newParticles[ip]->GetPDGMass());
    }


  if (numberOfSecondaries==2){
    // 2 -> 2
    
    //Get the CM energy
    G4double energy = cm4Momentum.e();
    G4ThreeVector p[2];

    //Size of momenta in CM
    

    // Energy conservation: 
    G4double cmMomentum = 1/(2*energy)*sqrt(energy*energy*energy*energy + m[0]*m[0]*m[0]*m[0] + m[1]*m[1]*m[1]*m[1]
					    - 2*(m[0]*m[0] + m[1]*m[1])*energy*energy -2*m[0]*m[0]*m[1]*m[1]);
    p[0] = cmMomentum * randomDirection;
    p[1] = -p[0];

    if(m_verboseLevel >= 2) std::cout << "ToyModelHadronicProcess::PostStepDoIt  2->2: " << std::endl 
				      << "     Pcm(GeV)   = " << cmMomentum / GeV << std::endl;

    for(int ip=0;ip<2;ip++)     
      {
	//Compute energy
	G4double e = sqrt(p[ip].mag2() + m[ip]*m[ip]);
	//Set 4-vectory
	fourMomenta.push_back(G4LorentzVector(p[ip],e));
	//Boost back to lab
	fourMomenta[ip].boost(cmBoost);      

	if(m_verboseLevel >= 2) std::cout  << "     particle " << ip <<" Plab(GeV)  = " << fourMomenta[ip] /GeV << std::endl;
      }

  } else if (numberOfSecondaries==3) {

    // 2 -> 3
    //Size of momenta in CM
    for (std::vector<G4double>::iterator it=m.begin();it!=m.end();it++) 
      fourMomenta.push_back(G4LorentzVector(0,0,0,*it));      

    Decay3Body KinCalc;
    KinCalc.doDecay(cm4Momentum, fourMomenta[0], fourMomenta[1], fourMomenta[2] );
    
    //Rotating the plane to a random orientation, and boosting home
    CLHEP::HepRotation rotation(randomDirection,G4UniformRand()*2*pi);
    for (std::vector<G4LorentzVector>::iterator it = fourMomenta.begin();
	 it!=fourMomenta.end();
	 it++)
      {
	*it *= rotation;
	(*it).boost(cmBoost);
      }
    if(m_verboseLevel >= 3) G4cout<<"Momentum-check: "<<incident4Momentum /GeV<<" GeV vs "
				  << (fourMomenta[0]+fourMomenta[1]+fourMomenta[2])/GeV<<G4endl;

  }
    
    
  //Now we have the fourMomenta of all the products (coming from 2->2 or 2->3)
  if(incidentSurvives){
    //if incident particle survives the number of secondaries is n-1
    m_particleChange.SetNumberOfSecondaries(numberOfSecondaries-1);
    if(m_verboseLevel >= 3) std::cout  << "Incident survives: set num secondaries to " << numberOfSecondaries-1 << std::endl;

  } else {
    //incident particle has to be killed and number of secondaries is n
    m_particleChange.SetNumberOfSecondaries(numberOfSecondaries);
    m_particleChange.ProposeTrackStatus(fStopAndKill);
    if(m_verboseLevel >= 3) std::cout  << "Incident does not survive: stopAndKill + set num secondaries to " << numberOfSecondaries << std::endl;
  }  
  //  double e_kin;

  for (int ip=0; ip <numberOfSecondaries;ip++)
    {
      if (newParticlesRHadron[ip]==incidentRHadron->GetDefinition()) // does incident paricle survive?
	{
	  //	incidentSurvives = true; //yes! Modify its dynamic properties
	  if(m_detachCloud) 
	    {
	      if(m_verboseLevel >= 3)
		std::cout  << "ToyModelHadronicProcess::PostStepDoIt   Add gluino momentum " <<
		  fourMomenta[ip]/GeV <<"(m="<< fourMomenta[ip].m()/ GeV<<") + " <<gluinoMomentum/GeV<<std::endl; ;
	      G4LorentzVector p4_new = gluinoMomentum+fourMomenta[ip];
	      G4ThreeVector momentum = p4_new.vect();
	      double rhMass=newParticlesRHadron[ip]->GetPDGMass() ;
	      //	      e_kin = sqrt(momentum.mag()*momentum.mag()+rhMass*rhMass)-rhMass;
	      //	      fourMomenta[ip]=G4LorentzVector(momentum,sqrt(momentum.mag2()+rhMass*rhMass));
	      fourMomenta[ip].setVectM(momentum,rhMass);

//	      double virt=(p4_new-fourMomenta[ip]).m()/MeV;

	      if(m_verboseLevel >= 3)
		std::cout <<  " = " << fourMomenta[ip]/GeV <<"(m="<< fourMomenta[ip].m() / GeV<<") vs. "<<rhMass/GeV 
			  << std::endl;     	
	    }
	  m_particleChange.ProposeMomentumDirection(fourMomenta[ip].vect()/fourMomenta[ip].vect().mag()); 
	  m_particleChange.ProposeEnergy(fourMomenta[ip].e()-fourMomenta[ip].mag());      
	  if(m_verboseLevel >= 3) std::cout  << "ToyModelHadronicProcess::PostStepDoIt   Propose momentum " << fourMomenta[ip]/GeV << std::endl;
	} else { //this particle is not the incident one
	  //Create new dynamic particle
	  G4DynamicParticle* productDynParticle = new G4DynamicParticle();
	  //Set the pDef to the dynParte
	  productDynParticle->SetDefinition(newParticlesRHadron[ip]);
	  //Set the 4-vector to dynPart
	  if(newParticlesRHadron[ip]!=newParticles[ip] && m_detachCloud )  // check if it is a cloud, second check is useless
	    {
	      G4LorentzVector p4_new;
	      p4_new.setVectM(gluinoMomentum.vect()+fourMomenta[ip].vect(),productDynParticle->GetDefinition()->GetPDGMass());
	      //	      productDynParticle->SetMomentum(fourMomenta[ip].vect()+gluinoMomentum.vect());  
	      productDynParticle->Set4Momentum(p4_new);  

//	      double virt=(gluinoMomentum+fourMomenta[ip]-p4_new).m()/MeV;

	      if(m_verboseLevel >= 3)
		std::cout  << "ToyModelHadronicProcess::PostStepDoIt   Add gluino momentum " <<
		  fourMomenta[ip]/GeV 
			   <<"(m="<< fourMomenta[ip].m()/ GeV<<") + " 
			   <<gluinoMomentum/GeV 
			   <<" = " << productDynParticle->Get4Momentum()/GeV
			   <<" => "<<productDynParticle->Get4Momentum().m()/GeV
			   <<" vs. "<<productDynParticle->GetDefinition()->GetPDGMass()/GeV
			   << std::endl;     	
	    }
	  else
	    {
	      productDynParticle->Set4Momentum(fourMomenta[ip]);       	
	    }
	  //Create a G4Track
	  G4Track* productTrack = new G4Track(productDynParticle,
					      track.GetGlobalTime(),
					      position);
      productTrack->SetTouchableHandle(thisTouchable);
	  //Append to the result
	  if(m_verboseLevel >= 3) std::cout  << "ToyModelHadronicProcess::PostStepDoIt   Add secondary with 4-Momentum " << productDynParticle->Get4Momentum()/GeV << std::endl;
	  m_particleChange.AddSecondary(productTrack);
	}
    } 

  //clear interaction length      
  ClearNumberOfInteractionLengthLeft();
  //return the result
  return &m_particleChange;
  
}

const G4DynamicParticle* ToyModelHadronicProcess::FindRhadron(G4ParticleChange* aParticleChange)
{
  G4int nsec = aParticleChange->GetNumberOfSecondaries();
  if (nsec==0) return 0;
  int i = 0;
  G4bool found = false;
  while (i!=nsec && !found){
    if (dynamic_cast<CustomParticle*>(aParticleChange->GetSecondary(i)->GetDynamicParticle()->GetDefinition())!=0) found = true;
    i++;
  }
  i--;
  if(found) return aParticleChange->GetSecondary(i)->GetDynamicParticle();
  return 0;
}
