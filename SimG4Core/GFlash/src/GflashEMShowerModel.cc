//
// ********************************************************************
// * DISCLAIMER                                                       *
// *                                                                  *
// * The following disclaimer summarizes all the specific disclaimers *
// * of contributors to this software. The specific disclaimers,which *
// * govern, are listed with their locations in:                      *
// *   http://cern.ch/geant4/license                                  *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.                                                             *
// *                                                                  *
// * This  code  implementation is the  intellectual property  of the *
// * GEANT4 collaboration.                                            *
// * By copying,  distributing  or modifying the Program (or any work *
// * based  on  the Program)  you indicate  your  acceptance of  this *
// * statement, and all its terms.                                    *
// ********************************************************************
//
//E.Barberio & Joanna Weng 

#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4Gamma.hh"
#include "G4NeutrinoE.hh"
#include "G4NeutrinoMu.hh"
#include "G4NeutrinoTau.hh"
#include "G4AntiNeutrinoE.hh"
#include "G4AntiNeutrinoMu.hh"
#include "G4AntiNeutrinoTau.hh"
#include "G4PionZero.hh"
#include "G4VProcess.hh"
#include "G4ios.hh"
#include "G4LogicalVolume.hh"
#include "geomdefs.hh"

#include "SimG4Core/GFlash/interface/GflashEMShowerModel.h"

#include "GFlashEnergySpot.hh"
#include "GFlashHomoShowerParameterisation.hh"
#include "GFlashSamplingShowerParameterisation.hh"

GflashEMShowerModel::GflashEMShowerModel(G4String modelName,
                                     G4Envelope* envelope)
  : G4VFastSimulationModel(modelName, envelope),
    PBound(0), Parameterisation(0), HMaker(0)
{
  G4cout << " --- Constructor GflashEMShowerModel ---" << G4endl;
	FlagParamType           = 1;
	FlagParticleContainment = 1;  
	StepInX0 = 0.1;
	Messenger       = new GflashEMShowerModelMessenger(this); 
	model_trigger=0;
	isapp=0;
	edoit=0;
}
// -----------------------------------------------------------------------------------

GflashEMShowerModel::GflashEMShowerModel(G4String modelName)
  : G4VFastSimulationModel(modelName),
    PBound(0), Parameterisation(0), HMaker(0)
{
	FlagParamType           =1;
	FlagParticleContainment = 1;  
	StepInX0 = 0.1; 
	Messenger       = new GflashEMShowerModelMessenger(this); 
}

// -----------------------------------------------------------------------------------

GflashEMShowerModel::~GflashEMShowerModel()
{
	Messenger       = new GflashEMShowerModelMessenger(this);
}

G4bool GflashEMShowerModel::IsApplicable(const G4ParticleDefinition& particleType)
{ 
	isapp++;  
	std::cout << "isapp =   " << isapp  <<std::endl;
	return 
	&particleType == G4Electron::ElectronDefinition() ||
	&particleType == G4Positron::PositronDefinition(); 
}
// -----------------------------------------------------------------------------------

/*********************************************************************/
/* Checks whether conditions of fast parametrisation  are fullfilled */
/**********************************************************************/
G4bool GflashEMShowerModel::ModelTrigger(const G4FastTrack & fastTrack )
{       
	model_trigger++;  
	std::cout << " model_trigger =   " << model_trigger<< "  " << fastTrack.GetPrimaryTrack()->GetKineticEnergy() /GeV  << std::endl;
	G4bool select = false;
	if(FlagParamType != 0)                  
	{  	test = fastTrack.GetPrimaryTrack()->GetMomentumDirection();
		
		// CMS eta check - no parameterisation between barrel and endcap
		G4double eta =   fastTrack.GetPrimaryTrack()->GetMomentum().pseudoRapidity() ;
		if ( (fabs(eta) > 1.45) &&  (fabs(eta) < 1.55) ) return false;
		G4double  ParticleEnergy = fastTrack.GetPrimaryTrack()->GetKineticEnergy(); 
		G4ParticleDefinition &ParticleType = *(fastTrack.GetPrimaryTrack()->GetDefinition());
		EnergyStop= PBound->GetEneToKill(ParticleType);	
		//	if ((ParticleEnergy /GeV < 0.02) &&   (HMaker->check(&fastTrack)))  return true;
		
		if(ParticleEnergy > PBound->GetMinEneToParametrise(ParticleType) &&
		ParticleEnergy < PBound->GetMaxEneToParametrise(ParticleType) )		{	
			select     = CheckParticleDefAndContainment(fastTrack);
			if (select ) {
				if ( HMaker->check(&fastTrack)) {		 //sensitive detector ?        
					///check conditions depending on particle flavour
					Parameterisation->GenerateLongitudinalProfile(ParticleEnergy); // performance to be optimized @@@@@@@
					EnergyStop= PBound->GetEneToKill(ParticleType);
				} //sens
				else select = false;
			} //comtai		  
		} // energy		
	}	
	test = fastTrack.GetPrimaryTrack()->GetMomentumDirection();
	return select; 

  return true;

}

// -----------------------------------------------------------------------------------
G4bool GflashEMShowerModel::CheckParticleDefAndContainment(const G4FastTrack& fastTrack)
{  
	G4bool filter=false;
	G4ParticleDefinition * ParticleType = fastTrack.GetPrimaryTrack()->GetDefinition(); 
	
	if(  ParticleType == G4Electron::ElectronDefinition() || 
	ParticleType == G4Positron::PositronDefinition() ){
	  filter=true;
	  if(FlagParticleContainment == 1)  {
	    filter=CheckContainment(fastTrack); 
	  }
	}
	return filter;  
}


// -----------------------------------------------------------------------------------
G4bool GflashEMShowerModel::CheckContainment(const G4FastTrack& fastTrack)
{
	//Note: typedef Hep3Vector G4ThreeVector;
	
	G4bool filter=false;
	//track informations
	G4ThreeVector DirectionShower = fastTrack.GetPrimaryTrackLocalDirection();
	G4ThreeVector InitialPositionShower = fastTrack.GetPrimaryTrackLocalPosition();
	
	G4ThreeVector OrthoShower, CrossShower; 
	//Returns orthogonal vector 
	OrthoShower = DirectionShower.orthogonal();
	// Shower in direction perpendicular to OrthoShower and DirectionShower
	CrossShower = DirectionShower.cross(OrthoShower);
	
	G4double  R     = Parameterisation->GetAveR90();
	G4double  Z     = Parameterisation->GetAveT90();
	G4int CosPhi[4] = {1,0,-1,0};
	G4int SinPhi[4] = {0,1,0,-1};
	
	G4ThreeVector Position;
	G4int NlateralInside=0;
	//pointer to soild we're in
	G4VSolid *SolidCalo = fastTrack.GetEnvelopeSolid();
	for(int i=0; i<4 ;i++){
		// polar coordinates
		Position = InitialPositionShower       + 
		Z*DirectionShower           +
		R*CosPhi[i]*OrthoShower     +
		R*SinPhi[i]*CrossShower     ;		
		if(SolidCalo->Inside(Position) != kOutside)
			NlateralInside=NlateralInside++;
	}
	
	//chose to parametrise or flag when all inetc...
	if(NlateralInside==4) filter=true;
	// std::cout << " points =   " <<NlateralInside << std::endl;
	return filter;
}
// -----------------------------------------------------------------------------------

void GflashEMShowerModel::DoIt(const G4FastTrack& fastTrack, G4FastStep& fastStep)
{
	// parametrise electrons
	if(fastTrack.GetPrimaryTrack()->GetDefinition() == G4Electron::ElectronDefinition() || 
		fastTrack.GetPrimaryTrack()->GetDefinition() == G4Positron::PositronDefinition() ) 
	ElectronDoIt(fastTrack,fastStep);
}
// -----------------------------------------------------------------------------------
void GflashEMShowerModel::ElectronDoIt(const G4FastTrack& fastTrack,  G4FastStep& fastStep)
{
	G4double ene=0;
	//********** try to speed up -> not good enough precision 
	//G4double  ParticleEnergy = fastTrack.GetPrimaryTrack()->GetKineticEnergy();
	//   if (ParticleEnergy /GeV< 0.02) 
	//     {
		
		//       //	 Kill the electron and deposit the energy using an exponential decay
		//       	  this->KillParticle( fastTrack, fastStep );
		//        	return;
	//     }
	//   ///***********
	
	std::cout<<"--- GflashEMShowerModel: ElectronDoit ---"<< " "<< fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV << std::endl;
	if (test != fastTrack.GetPrimaryTrack()->GetMomentumDirection()) {
		//	std::cout<< fastTrack.GetPrimaryTrack() << std::endl;
		//	std::cout<< " !!!!!!!!!!!!  ERROR  FOUND !!!!!!!!!  test ="  << test << " eledoit "<< fastTrack.GetPrimaryTrack()->GetMomentumDirection()  << std::endl;   
        }
	fastStep.KillPrimaryTrack(); 
	fastStep.SetPrimaryTrackPathLength(0.0);
	fastStep.SetTotalEnergyDeposited(fastTrack.GetPrimaryTrack()->GetKineticEnergy());
	
	
	G4double Energy =fastTrack.GetPrimaryTrack()->GetKineticEnergy(); 
	// Correction for possible G4 Error
	G4ThreeVector DirectionShower =test;
	G4ThreeVector OrthoShower, CrossShower ;
	OrthoShower = DirectionShower.orthogonal();
	CrossShower = DirectionShower.cross(OrthoShower);
	//--------------------------------
	///Generate longitudinal profile
	//--------------------------------
	Parameterisation->GenerateLongitudinalProfile(Energy); // performane iteration @@@@@@@	
	///Initialisation of long. loop variables
	G4VSolid *SolidCalo = fastTrack.GetEnvelopeSolid();
	G4ThreeVector pos   = fastTrack.GetPrimaryTrackLocalPosition();
	G4ThreeVector dir   = fastTrack.GetPrimaryTrackLocalDirection();
	G4double Bound      = SolidCalo->DistanceToOut(pos,dir); 
	
	G4double Dz       = 0.00;     
	G4double ZEndStep = 0.00;
	
	G4double EnergyNow        = Energy;
	G4double EneIntegral      = 0.00;   
	G4double LastEneIntegral  = 0.00;   
	G4double DEne             = 0.00;	
	G4double NspIntegral      = 0.00;   
	G4double LastNspIntegral  = 0.00;   
	G4double DNsp             = 0.00;
	
	// starting point of the shower:
	G4ThreeVector PositionShower  = fastTrack.GetPrimaryTrack()->GetPosition();
	G4ThreeVector NewPositionShower    = PositionShower;   
	G4double      StepLenght           = 0.00;	
	G4int NSpotDeposited =0;
	
	//--------------------------
	/// Begin Longitudinal Loop
	//-------------------------	
	do
	{  
		//determine step size=min(1Xo,next boundary)
		G4double stepLength = StepInX0*Parameterisation->GetX0();
		if(Bound < stepLength){ 
			Dz    = Bound;
			Bound = 0.00;
		}
		else{ 
			Dz    = stepLength;
			Bound = Bound-Dz;
		}
		ZEndStep=ZEndStep+Dz;		
		// Determine Energy Release in Step
		if(EnergyNow > EnergyStop){
			LastEneIntegral  = EneIntegral;
			EneIntegral      = Parameterisation->IntegrateEneLongitudinal(ZEndStep);
			DEne             = std::min( EnergyNow, (EneIntegral-LastEneIntegral)*Energy);
			LastNspIntegral  = NspIntegral;
			NspIntegral      = Parameterisation->IntegrateNspLongitudinal(ZEndStep);
			DNsp             = std::max(1., std::floor( (NspIntegral-LastNspIntegral)*Parameterisation->GetNspot() ) );
		}
		// end of the shower
		else{    
			DEne             = EnergyNow;
			DNsp             = std::max(1., std::floor( (1.- NspIntegral)*Parameterisation->GetNspot() ));
		} 
		EnergyNow  = EnergyNow - DEne;
		
		//move particle in the middle of the step
		StepLenght        = StepLenght + Dz/2.00;  
		NewPositionShower = NewPositionShower + 
		StepLenght*DirectionShower;
		StepLenght        = Dz/2.00;
		GFlashEnergySpot Spot;     
		//protection against endless loops 
		int security_bound = 5;
		int security = 0;
		//generate spots & hits:
		for (int i = 0; i < DNsp; i++)
		{ 
			NSpotDeposited=NSpotDeposited++;		      						
			//Spot energy: the same for all spots
			Spot.SetEnergy( DEne / DNsp );
			G4double PhiSpot = Parameterisation->GeneratePhi(); // phi of spot
			G4double RSpot   = Parameterisation->GenerateRadius(i,Energy,ZEndStep-Dz/2.); // radius of spot	
			G4ThreeVector SpotPosition = NewPositionShower                         +
			Dz/DNsp*DirectionShower*(i+1/2.-DNsp/2.)  +
			RSpot*std::cos(PhiSpot)*OrthoShower            +  
			RSpot*std::sin(PhiSpot)*CrossShower;      
			Spot.SetPosition(SpotPosition);			
			//Generate Hits of this spot      			

			if (!HMaker->makesensitive(&Spot, &fastTrack)) 
			{
				--i ;
				security++;
			}
			if( security > security_bound) 
			{
				ene+= Spot.GetEnergy();
				HMaker->make(&Spot, &fastTrack);
				i++;
			}			
		}       
	}
	while(EnergyNow > 0.0 && Bound> 0.0);     
	//---------------
	/// End Loop
	//------------- 	
	std::cout <<"SYJUN  Energy Lost " <<  ene/GeV  << std::endl;;	
}
// -----------------------------------------------------------------------------------

// void GflashEMShowerModel::KillParticle(const G4FastTrack& fastTrack, G4FastStep& fastStep)
// {
	
// 	// Kill the particle to be parametrised
// 	fastStep.KillPrimaryTrack();
// 	fastStep.SetPrimaryTrackPathLength(0.0);
// 	fastStep.SetTotalEnergyDeposited(fastTrack.GetPrimaryTrack()->GetKineticEnergy());
// 	GFlashEnergySpot Spot;  
// 	G4double Energy = fastTrack.GetPrimaryTrack()->GetKineticEnergy();
// 	// axis of the shower, in global reference frame:
// 	G4ThreeVector DirectionShower =test;
// 	// starting point of the shower:
// 	G4ThreeVector PositionShower = fastTrack.GetPrimaryTrack()->GetPosition();
// 	;
// 	// Generate the exponential decay of the energy
// 	//  G4double dist = Parametrisation->GenerateExponential(Energy); 
// 	Spot.SetEnergy( Energy);
// 	//  G4ThreeVector SpotPosition = PositionShower + dist * DirectionShower;  
// 	Spot.SetPosition(PositionShower);     
// 	HMaker->make(&Spot, &fastTrack);
// }
