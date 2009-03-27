//
// $Id: GflashEMShowerProfile.cc,v 1.15 2009/03/24 03:48:55 syjun Exp $
// initial setup : Soon Jun & Dongwook Jang
// Translated from Fortran code.

#include "Randomize.hh"
#include "G4TransportationManager.hh"
#include "G4VPhysicalVolume.hh" 
#include "G4LogicalVolume.hh"
#include "G4VSensitiveDetector.hh"
#include "G4EventManager.hh"
#include "G4SteppingManager.hh"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/GenericFunctions/IncompleteGamma.hh"
#include "FWCore/Utilities/interface/Exception.h"

#include "SimG4Core/Application/interface/SteppingAction.h"
#include "SimG4Core/GFlash/interface/GflashEMShowerProfile.h"
#include "SimG4Core/GFlash/interface/GflashHistogram.h"
#include "SimG4Core/GFlash/interface/GflashTrajectory.h"
#include "SimG4Core/GFlash/interface/GflashTrajectoryPoint.h"

GflashEMShowerProfile::GflashEMShowerProfile(G4Region* envelope, edm::ParameterSet parSet) : theParSet(parSet)
{
  theHelix = new GflashTrajectory;
  theGflashStep = new G4Step();
  theGflashNavigator = 0;
  theGflashTouchableHandle = new G4TouchableHistory();

  theHisto = GflashHistogram::instance();

  jCalorimeter = Gflash::kNULL;
  theBField = parSet.getParameter<double>("bField");
  theWatcherOn = parSet.getParameter<bool>("watcherOn");

  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "GflashHadronShowerProfile requires RandomNumberGeneratorService\n"
      << "which is not present in the configuration file. "
      << "You must add the service\n in the configuration file or "
      << "remove the modules that require it.";
  }
  theRandGauss = new CLHEP::RandGaussQ(rng->getEngine());

  theTuning_pList = parSet.getParameter<std::vector<double> >("tuning_pList");

}


GflashEMShowerProfile::~GflashEMShowerProfile()
{
  delete theHelix;
  delete theRandGauss;
  if(theGflashStep) delete theGflashStep;
}


void GflashEMShowerProfile::parameterization(const G4FastTrack& fastTrack)
{

  // This part of code is copied from the original GFlash Fortran code.
  // reference : hep-ex/0001020v1

  const G4double energyCutoff     = 0.01; 
  const G4int    maxNumberOfSpots = 100000;

   G4ThreeVector showerStartingPosition = fastTrack.GetPrimaryTrack()->GetPosition() / cm;
  G4ThreeVector showerMomentum = fastTrack.GetPrimaryTrack()->GetMomentum()/GeV;

  //find the calorimeter at the shower starting point
  jCalorimeter = Gflash::getCalorimeterNumber(showerStartingPosition);

  G4double incomingEnergy   = fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV;
  G4double logEinc = std::log(incomingEnergy);
  G4double y = incomingEnergy / Gflash::criticalEnergy; // y = E/Ec, criticalEnergy is in GeV
  G4double logY = std::log(y);

  G4double nSpots = 93.0 * std::log(Gflash::Z[jCalorimeter]) * std::pow(incomingEnergy,0.876); // total number of spots

  // implementing magnetic field effects
  double charge = fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint()->GetCharge();
  theHelix->initializeTrajectory(showerMomentum,showerStartingPosition,charge,theBField);

  //path Length from the origin to the shower starting point in cm
  G4double pathLength0 = theHelix->getPathLengthAtRhoEquals(showerStartingPosition.getRho());
  G4double pathLength = pathLength0; // this will grow along the shower development

  //--- 2.2  Fix intrinsic properties of em. showers.

  G4double fluctuatedTmax = std::log(logY - 0.7157);
  G4double fluctuatedAlpha = std::log(0.7996 +(0.4581 + 1.8628/Gflash::Z[jCalorimeter])*logY);

  G4double sigmaTmax = 1.0/( -1.4  + 1.26 * logY);
  G4double sigmaAlpha = 1.0/( -0.58 + 0.86 * logY);
  G4double rho = 0.705  - 0.023 * logY;
  G4double sqrtPL = std::sqrt((1.0+rho)/2.0);
  G4double sqrtLE = std::sqrt((1.0-rho)/2.0);

  G4double norm1 = theRandGauss->fire();
  G4double norm2 = theRandGauss->fire();
  G4double tempTmax = fluctuatedTmax + sigmaTmax*(sqrtPL*norm1 + sqrtLE*norm2);
  G4double tempAlpha = fluctuatedAlpha + sigmaAlpha*(sqrtPL*norm1 - sqrtLE*norm2);

  // tmax, alpha, beta : parameters of gamma distribution
  G4double tmax = std::exp(tempTmax);
  G4double alpha = std::exp(tempAlpha);
  G4double beta = (alpha - 1.0)/tmax;
 
  // spot fluctuations are added to tmax, alpha, beta
  G4double averageTmax = logY-0.858;
  G4double averageAlpha = 0.21+(0.492+2.38/Gflash::Z[jCalorimeter])*logY;
  G4double spotTmax  = averageTmax * (0.698 + .00212*Gflash::Z[jCalorimeter]);
  G4double spotAlpha = averageAlpha * (0.639 + .00334*Gflash::Z[jCalorimeter]);
  G4double spotBeta = (spotAlpha-1.0)/spotTmax;


   if(theHisto->getStoreFlag()) {
    theHisto->em_incE->Fill(incomingEnergy);
    theHisto->em_ssp_rho->Fill(showerStartingPosition.rho());
    theHisto->em_ssp_z->Fill(showerStartingPosition.z());
  }

  //  parameters for lateral distribution and fluctuation
  G4double z1=0.0251+0.00319*logEinc;
  G4double z2=0.1162-0.000381*Gflash::Z[jCalorimeter];

  G4double k1=0.659 - 0.00309 * Gflash::Z[jCalorimeter];
  G4double k2=0.645;
  G4double k3=-2.59;
  G4double k4=0.3585+ 0.0421*logEinc;

  G4double p1=2.623 -0.00094*Gflash::Z[jCalorimeter];
  G4double p2=0.401 +0.00187*Gflash::Z[jCalorimeter];
  G4double p3=1.313 -0.0686*logEinc;

  // @@@ dwjang, intial tuning by comparing 20-150GeV TB data
  // the width of energy response is not yet tuned.
  G4double e25Scale = 1.007;
  p1 *= 0.96;

  // preparation of longitudinal integration
  G4double stepLengthLeft = fastTrack.GetEnvelopeSolid()->DistanceToOut(fastTrack.GetPrimaryTrackLocalPosition(),
									fastTrack.GetPrimaryTrackLocalDirection()) / cm;
  G4int    nSpots_sd = 0; // count total number of spots in SD
  G4double zInX0 = 0.0; // shower depth in X0 unit
  G4double deltaZInX0 = 0.0; // segment of depth in X0 unit
  G4double deltaZ = 0.0; // segment of depth in cm
  G4double stepLengthLeftInX0 = 0.0; // step length left in X0 unit

  const G4double divisionStepInX0 = 0.1; //step size in X0 unit
  G4double energy = incomingEnergy; // energy left in GeV

  Genfun::IncompleteGamma gammaDist;

  G4double energyInGamma = 0.0; // energy in a specific depth(z) according to Gamma distribution
  G4double preEnergyInGamma = 0.0; // energy calculated in a previous depth
  G4double sigmaInGamma  = 0.; // sigma of energy in a specific depth(z) according to Gamma distribution
  G4double preSigmaInGamma = 0.0; // sigma of energy in a previous depth

  //energy segment in Gamma distribution of shower in each step  
  G4double deltaEnergy =0.0 ; // energy in deltaZ
  G4int spotCounter = 0; // keep track of number of spots generated

  //step increment along the shower direction
  G4double deltaStep = 0.0;


  // The time is not meaningful but G4Step requires that information to make a step unique.
  // Uniqueness of G4Step is important otherwise hits won't be created.
  G4double timeGlobal = fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint()->GetGlobalTime();

  // this needs to be deleted manually at the end of this loop.
  theGflashNavigator = new G4Navigator();
  theGflashNavigator->SetWorldVolume(G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume());

  // loop for longitudinal integration
  while(energy > 0.0 && stepLengthLeft > 0.0) { 

    stepLengthLeftInX0 = stepLengthLeft / Gflash::radLength[jCalorimeter];

    if ( stepLengthLeftInX0 < divisionStepInX0 ) {
      deltaZInX0 = stepLengthLeftInX0;
      deltaZ     = deltaZInX0 * Gflash::radLength[jCalorimeter];
      stepLengthLeft = 0.0;
    }
    else {
      deltaZInX0 = divisionStepInX0;
      deltaZ     = deltaZInX0 * Gflash::radLength[jCalorimeter];
      stepLengthLeft -= deltaZ;
    }

    zInX0 += deltaZInX0;


    G4int nSpotsInStep = 0;

    if ( energy > energyCutoff  ) {
      preEnergyInGamma  = energyInGamma;
      gammaDist.a().setValue(alpha);  //alpha
      energyInGamma = gammaDist(beta*zInX0); //beta
      G4double energyInDeltaZ  = energyInGamma - preEnergyInGamma;
      deltaEnergy   = std::min(energy,incomingEnergy*energyInDeltaZ);
 
      preSigmaInGamma  = sigmaInGamma;
      gammaDist.a().setValue(spotAlpha);  //alpha spot
      sigmaInGamma = gammaDist(spotBeta*zInX0); //beta spot
      nSpotsInStep = std::max(1,int(nSpots * (sigmaInGamma - preSigmaInGamma)));
    }
    else {
      deltaEnergy = energy;
      preSigmaInGamma  = sigmaInGamma;
      nSpotsInStep = std::max(1,int(nSpots * (1.0 - preSigmaInGamma)));
    }

    if ( deltaEnergy > energy || (energy-deltaEnergy) < energyCutoff ) deltaEnergy = energy;

    energy  -= deltaEnergy;

    if ( spotCounter+nSpotsInStep > maxNumberOfSpots ) {
      nSpotsInStep = maxNumberOfSpots - spotCounter;
      if ( nSpotsInStep < 1 ) { // @@ check
        std::cout << "GflashEMShowerProfile::Parameterization : Too Many Spots " << std::endl;
        std::cout << "                       break to regenerate nSpotsInStep " << std::endl;
        break;
      }
    }


    // It begins with 0.5 of deltaZ and then icreases by 1 deltaZ
    deltaStep  += 0.5*deltaZ;
    pathLength += deltaStep;
    deltaStep   =  0.5*deltaZ;


    // lateral shape and fluctuations for  homogenous calo.
    G4double tScale = tmax *alpha/(alpha-1.0) * (std::exp(fluctuatedAlpha)-1.0)/std::exp(fluctuatedAlpha);
    G4double tau = std::min(10.0,(zInX0 - 0.5*deltaZInX0)/tScale);
    G4double rCore = z1 + z2 * tau; 
    G4double rTail = k1 *( std::exp(k3*(tau-k2)) + std::exp(k4*(tau-k2))); // @@ check RT3 sign
    G4double p23 = (p2 - tau)/p3;
    G4double probabilityWeight = p1 *  std::exp( p23 - std::exp(p23) );


    // Deposition of spots according to lateral distr.
    // Apply absolute energy scale
    // Convert into MeV unit
    G4double emSpotEnergy = deltaEnergy / nSpotsInStep * e25Scale * GeV;

    for (G4int ispot = 0 ;  ispot < nSpotsInStep ; ispot++) {
      spotCounter++;

      G4double u1 = G4UniformRand();
      G4double u2 = G4UniformRand();
      G4double rInRM = 0.0;

      if (u1 < probabilityWeight) {
	rInRM = rCore * std::sqrt( u2/(1.0-u2) );
      }
      else {
	rInRM = rTail * std::sqrt( u2/(1.0-u2) );
      }

      G4double rShower =  rInRM * Gflash::rMoliere[jCalorimeter];

      // Uniform & random rotation of spot along the azimuthal angle
      G4double azimuthalAngle = twopi*G4UniformRand();

      // Compute global position of generated spots with taking into account magnetic field
      // Divide deltaZ into nSpotsInStep and give a spot a global position
      G4double incrementPath = (deltaZ/nSpotsInStep)*(ispot+0.5 - 0.5*nSpotsInStep);

      // trajectoryPoint give a spot an imaginary point along the shower development
      GflashTrajectoryPoint trajectoryPoint;
      theHelix->getGflashTrajectoryPoint(trajectoryPoint,pathLength+incrementPath);

      // actual spot position by adding a radial vector to a trajectoryPoint
      G4ThreeVector SpotPosition = trajectoryPoint.getPosition() +
	rShower*std::cos(azimuthalAngle)*trajectoryPoint.getOrthogonalUnitVector() +
	rShower*std::sin(azimuthalAngle)*trajectoryPoint.getCrossUnitVector();

      // Convert into mm unit
      SpotPosition *= cm;

      //---------------------------------------------------
      // fill a fake step to send it to hit maker
      //---------------------------------------------------

      // to make a different time for each fake step. (0.03 nsec is corresponding to 1cm step size)
      timeGlobal += 0.0001*nanosecond;

      // fill equivalent changes to a (fake) step associated with a spot 

      theGflashStep->SetTrack(const_cast<G4Track*>(fastTrack.GetPrimaryTrack()));
      theGflashStep->GetPostStepPoint()->SetGlobalTime(timeGlobal);
      theGflashStep->GetPreStepPoint()->SetPosition(SpotPosition);
      theGflashStep->GetPostStepPoint()->SetPosition(SpotPosition);
      theGflashStep->GetPostStepPoint()->SetProcessDefinedStep(const_cast<G4VProcess*> (fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()));

      //put touchable for each energy spot so that touchable history keeps track of each step.
      theGflashNavigator->LocateGlobalPointAndUpdateTouchableHandle(SpotPosition,G4ThreeVector(0,0,0),theGflashTouchableHandle, false);
      theGflashStep->GetPreStepPoint()->SetTouchableHandle(theGflashTouchableHandle);
      theGflashStep->SetTotalEnergyDeposit(emSpotEnergy);
    
      // if there is a watcher defined in a job and the flag is turned on
      if(theWatcherOn) {
	SteppingAction* userSteppingAction = (SteppingAction*) G4EventManager::GetEventManager()->GetUserSteppingAction();
	userSteppingAction->m_g4StepSignal(theGflashStep);
      }

      G4double zInX0_spot = std::abs(pathLength+incrementPath - pathLength0)/Gflash::radLength[jCalorimeter];

      if(theHisto->getStoreFlag()) {
	theHisto->em_long->Fill(zInX0_spot,emSpotEnergy/GeV);
	theHisto->em_lateral->Fill(zInX0_spot,rShower/Gflash::rMoliere[jCalorimeter],emSpotEnergy/GeV);
      }

      // Send G4Step information to Hit/Digi if the volume is sensitive
      // Copied from G4SteppingManager.cc
    
      G4VPhysicalVolume* aCurrentVolume = theGflashStep->GetPreStepPoint()->GetPhysicalVolume();
      if( aCurrentVolume == 0 ) continue;

      G4LogicalVolume* lv = aCurrentVolume->GetLogicalVolume();
      if(lv->GetRegion()->GetName() != "CaloRegion") continue;

      theGflashStep->GetPreStepPoint()->SetSensitiveDetector(aCurrentVolume->GetLogicalVolume()->GetSensitiveDetector());
      G4VSensitiveDetector* aSensitive = theGflashStep->GetPreStepPoint()->GetSensitiveDetector();
      
      if( aSensitive == 0 || (std::fabs(SpotPosition.getZ()/cm) > Gflash::Zmax[Gflash::kHE]) ) continue;
      aSensitive->Hit(theGflashStep);

      nSpots_sd++;

      // for histogramming      
      if(theHisto->getStoreFlag()) {
	theHisto->em_long_sd->Fill(zInX0_spot,emSpotEnergy/GeV);
	theHisto->em_lateral_sd->Fill(zInX0_spot,rShower/Gflash::rMoliere[jCalorimeter],emSpotEnergy/GeV);
      }
      
    } // end of for spot iteration

  } // end of while for longitudinal integration

  if(theHisto->getStoreFlag()) {
    theHisto->em_nSpots_sd->Fill(nSpots_sd);
  }

  delete theGflashNavigator;

}

