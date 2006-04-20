#include "SimG4Core/CustomPhysics/interface/GenericHadronicProcess.h"
#include "SimG4Core/CustomPhysics/interface/HadronicProcessHelper.h"
#include "SimG4Core/CustomPhysics/interface/Decay3Body.h"

#include "G4ProcessManager.hh"
#include "G4ParticleTable.hh"
#include "G4InelasticInteraction.hh"

GenericHadronicProcess::GenericHadronicProcess(const std::string & processName) :
    G4HadronicProcess(processName), m_verboseLevel(0)
{
    // Instantiating helper class
    m_helper = HadronicProcessHelper::instance();
}

  
bool GenericHadronicProcess::IsApplicable(const G4ParticleDefinition & aP)
{ return m_helper->applicabilityTester(aP); }

double GenericHadronicProcess::GetMicroscopicCrossSection(const G4DynamicParticle * particle,
							  const G4Element * element,
							  double /*temperature*/)
{
    // Get the cross section for this particle/element combination from the ProcessHelper
    double inclusiveCrossSection = m_helper->inclusiveCrossSection(particle,element);

    // Need to provide Set-methods for these in time
    double highestEnergyLimit = 10 * TeV  ;
    double lowestEnergyLimit = 1 * eV;
    double particleEnergy = particle->GetKineticEnergy();
   
    if (particleEnergy > highestEnergyLimit || particleEnergy < lowestEnergyLimit)
    {
	if(m_verboseLevel >= 1) 
	    std::cout << "GenericHadronicProcess: Energy out of bounds [" 
		      << lowestEnergyLimit / MeV << "MeV , "
		      << highestEnergyLimit / MeV << "MeV ] while it is " 
		      << particleEnergy/MeV  << std::endl;
	return 0;
    } 
    else 
    {
	if(m_verboseLevel >= 3) 
	    std::cout << "GenericHadronicProcess: Return cross section " 
		      << inclusiveCrossSection << std::endl;
	return inclusiveCrossSection;
    }
}

G4VParticleChange * GenericHadronicProcess::PostStepDoIt(const G4Track& track,
							 const G4Step& /*  step*/)
{
    // A little setting up
    m_particleChange.Initialize(track);
    const G4DynamicParticle * incidentParticle = track.GetDynamicParticle();
    const G4ThreeVector position = track.GetPosition();
    const G4int incidentParticlePDG = incidentParticle->GetDefinition()->GetPDGEncoding();
    G4ParticleTable * theParticleTable = G4ParticleTable::GetParticleTable();
    std::vector<G4ParticleDefinition*> newParticles;
    std::vector<G4DynamicParticle*> newDynamicParticles;
    // These are probably redundant, but they can easily be removed :-)

    bool incidentSurvives = false;
  
    // Get the final state particles and target
    G4ParticleDefinition * targetParticle; 
    HadronicProcessHelper::ReactionProduct reactionProduct = m_helper->finalState(track,targetParticle);
    
    // Fill a list of the new particles to create 
    // (i.e. reaction products without the incident if it survives)
    for(HadronicProcessHelper::ReactionProduct::iterator it  = reactionProduct.begin();
        it != reactionProduct.end(); it++)
    {
	G4ParticleDefinition * productDefinition =theParticleTable->FindParticle(*it);
	if (productDefinition->GetPDGEncoding()==incidentParticlePDG)
	    incidentSurvives = true;       
	newParticles.push_back(productDefinition);
    }
    
    int numberOfSecondaries = reactionProduct.size();

    if (m_verboseLevel >= 2) 
	std::cout << "GenericHadronicProcess::PostStepDoIt  N secondaries: " 
		  << numberOfSecondaries << std::endl;
  

    //************ My toy model ********************
    // 2 -> 2 goes to CM, chooses a random direction and fires the particles off back to back
    // 2 -> 3 Effectively two two-body decays

    // Getting four momenta
    const G4LorentzVector incident4Momentum = incidentParticle->Get4Momentum();
    const G4LorentzVector target4Momentum(0,0,0,targetParticle->GetPDGMass());
    const G4LorentzVector sum4Momentum = incident4Momentum + target4Momentum;
    const G4ThreeVector   cmBoost = sum4Momentum.boostVector();//The boost from CM to lab
    const G4LorentzVector cm4Momentum = sum4Momentum.rest4Vector();

    if (m_verboseLevel >= 2) 
	std::cout << "GenericHadronicProcess::PostStepDoIt  Kinematics in GeV: " << std::endl 
		  << "     Boost   = " << cmBoost / GeV<< std::endl
		  << "     4P CM   = " << cm4Momentum / GeV << std::endl
		  << "     4P Inc  = " << incident4Momentum / GeV << std::endl
		  << "     4P Targ = " << target4Momentum  / GeV<< std::endl;

    // Choosing random direction
    const double phi_p = 2*pi*RandFlat::shoot()-pi ;
    const double theta_p = pi*RandFlat::shoot() ;
    const G4ThreeVector randomDirection(sin(theta_p)*cos(phi_p),
					sin(theta_p)*sin(phi_p),
					cos(theta_p));
  
    std::vector<double> m;
    std::vector<G4LorentzVector> fourMomenta;
  
    // Fill the masses
    for (int ip=0;ip<numberOfSecondaries;ip++)     
    {
	m.push_back(newParticles[ip]->GetPDGMass());
    }	

    if (numberOfSecondaries==2)
    {
	// 2 -> 2
    
	// Get the CM energy
	double energy = cm4Momentum.e();
	G4ThreeVector p[2];
	// Size of momenta in CM
    
	// Energy conservation: 
	double cmMomentum = 1/(2*energy)*sqrt(energy*energy*energy*energy + 
					      m[0]*m[0]*m[0]*m[0] + m[1]*m[1]*m[1]*m[1] - 
					      2*(m[0]*m[0] + m[1]*m[1])*energy*energy -
					      2*m[0]*m[0]*m[1]*m[1]);
	p[0] = cmMomentum * randomDirection;
	p[1] = -p[0];

	if (m_verboseLevel >= 2) 
	    std::cout << "GenericHadronicProcess::PostStepDoIt  2->2: " << std::endl 
		      << "     Pcm(GeV)   = " << cmMomentum / GeV << std::endl;

	for (int ip=0;ip<2;ip++)     
	{
	    // Compute energy
	    double e = sqrt(p[ip].mag2() + m[ip]*m[ip]);
	    // Set 4-vectory
	    fourMomenta.push_back(G4LorentzVector(p[ip],e));
	    // Boost back to lab
	    fourMomenta[ip].boost(cmBoost);      
	    if (m_verboseLevel >= 2) 
		std::cout  << "     particle " << ip <<" Plab(GeV)  = " 
			   << fourMomenta[ip] /GeV << std::endl;
	}
    } 
    else if (numberOfSecondaries==3) 
    {
	// 2 -> 3
	// Size of momenta in CM
	for (std::vector<G4double>::iterator it=m.begin();it!=m.end();it++) 
	    fourMomenta.push_back(G4LorentzVector(0,0,0,*it));      

	Decay3Body KinCalc;
	KinCalc.doDecay(cm4Momentum, fourMomenta[0], fourMomenta[1], fourMomenta[2] );
    
	//Rotating the plane to a random orientation, and boosting home
	HepRotation rotation(randomDirection,RandFlat::shoot()*2*pi);
	for (std::vector<G4LorentzVector>::iterator it = fourMomenta.begin();
	     it!=fourMomenta.end(); it++)
	{
	    *it *= rotation;
	    (*it).boost(cmBoost);
	}
	if (m_verboseLevel >= 3) 
	    std::cout << "Momentum-check: " <<incident4Momentum /GeV << " GeV vs "
		      << (fourMomenta[0]+fourMomenta[1]+fourMomenta[2])/GeV << std::endl;

    }	
    
    // Now we have the fourMomenta of all the products (coming from 2->2 or 2->3)
    if (incidentSurvives)
    {
	// if incident particle survives the number of secondaries is n-1
	m_particleChange.SetNumberOfSecondaries(numberOfSecondaries-1);
	if (m_verboseLevel >= 3) 
	    std::cout  << "Incident survives: set num secondaries to " 
		       << numberOfSecondaries-1 << std::endl;

    } 
    else 
    {
	// incident particle has to be killed and number of secondaries is n
	m_particleChange.SetNumberOfSecondaries(numberOfSecondaries);
	m_particleChange.ProposeTrackStatus(fStopAndKill);
	if(m_verboseLevel >= 3) 
	    std::cout  << "Incident does not survive: stopAndKill + set num secondaries to " 
		       << numberOfSecondaries << std::endl;
    }  
    
    for (int ip=0; ip<numberOfSecondaries;ip++)
    {
	if (newParticles[ip]->GetPDGEncoding()==incidentParticlePDG) // does incident paricle survive?
	{
	    incidentSurvives = true; // yes! Modify its dynamic properties
	    m_particleChange.ProposeMomentumDirection
		(fourMomenta[ip].vect()/fourMomenta[ip].vect().mag()); 
	    m_particleChange.ProposeEnergy(fourMomenta[ip].e()-fourMomenta[ip].mag());      
	    if(m_verboseLevel >= 3) 
		std::cout  << "GenericHadronicProcess::PostStepDoIt   Propose momentum " 
			   << fourMomenta[ip]/GeV << std::endl;
	} 
	else 
	{ 
	    // this particle is not the incident one
	    // Create new dynamic particle
	    G4DynamicParticle * productDynParticle = new G4DynamicParticle();
	    // Set the pDef to the dynParte
	    productDynParticle->SetDefinition(newParticles[ip]);
	    // Set the 4-vector to dynPart
	    productDynParticle->Set4Momentum(fourMomenta[ip]);       	
	    //Create a G4Track
	    G4Track * productTrack = new G4Track(productDynParticle,track.GetGlobalTime(),position);
	    // Append to the result
	    if (m_verboseLevel >= 3) 
		std::cout  << "GenericHadronicProcess::PostStepDoIt   Add secondary with 4-Momentum " 
			   << fourMomenta[ip]/GeV << std::endl;
	    m_particleChange.AddSecondary(productTrack);
	}
    } 

    // clear interaction length      
    ClearNumberOfInteractionLengthLeft();
    // return the result
    return &m_particleChange; 
}
