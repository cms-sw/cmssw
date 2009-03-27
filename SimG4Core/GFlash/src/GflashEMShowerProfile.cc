//
// $Id: GflashEMShowerProfile.cc,v 1.3 2008/02/29 23:40:56 syjun Exp $
// initial setup : Soon Jun & Dongwook Jang
// Translated from Fortran code.

#include "Randomize.hh"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/GenericFunctions/IncompleteGamma.hh"
#include "FWCore/Utilities/interface/Exception.h"

#include "SimG4Core/GFlash/interface/GflashEMShowerProfile.h"
#include "SimG4Core/GFlash/interface/GflashEnergySpot.h"
#include "SimG4Core/GFlash/interface/GflashHistogram.h"
#include "SimG4Core/GFlash/interface/GflashTrajectory.h"
#include "SimG4Core/GFlash/interface/GflashTrajectoryPoint.h"

GflashEMShowerProfile::GflashEMShowerProfile(G4Region* envelope)
{
  theHelix = new GflashTrajectory;
  theHisto = GflashHistogram::instance();

  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "GflashHadronShowerProfile requires RandomNumberGeneratorService\n"
      << "which is not present in the configuration file. "
      << "You must add the service\n in the configuration file or "
      << "remove the modules that require it.";
  }
  theRandGauss = new CLHEP::RandGaussQ(rng->getEngine());
}

GflashEMShowerProfile::~GflashEMShowerProfile()
{
  delete theHelix;
  delete theRandGauss;
}

void GflashEMShowerProfile::parameterization(const G4FastTrack& fastTrack)
{
  // This part of code is copied from the original GFlash Fortran code.
  // reference : hep-ex/0001020v1

  const G4double energyCutoff     = 0.01; 
  const G4int    maxNumberOfSpots = 100000;

  G4double incomingEnergy   = fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV;
  const G4double radLength = 0.89; // cm
  const G4double Z = 68.360;
  const G4double rMoliere = 2.19; // cm
  const G4double criticalEnergy = 8.6155/GeV; // eScale*radLength/rMoliere (in MeV) need to convert into GeV
  G4double nSpots = 93.0 * std::log(Z) * std::pow(incomingEnergy,0.876); // total number of spots

  G4double logEinc = std::log(incomingEnergy);
  G4double y = incomingEnergy / criticalEnergy; // y = E/Ec, criticalEnergy is in GeV
  G4double logY = std::log(y);
 
  G4ThreeVector showerStartingPosition = fastTrack.GetPrimaryTrack()->GetPosition() / cm;
  G4ThreeVector showerMomentum = fastTrack.GetPrimaryTrack()->GetMomentum()/GeV;


  // implementing magnetic field effects
  const G4double bField = 4.0; // in Tesla
  double charge = fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint()->GetCharge();
  //  GflashTrajectory helix(showerMomentum,showerStartingPosition,charge,bField);
  theHelix->initializeTrajectory(showerMomentum,showerStartingPosition,charge,bField);

  //path Length from the origin to the shower starting point in cm
  G4double pathLength0 = theHelix->getPathLengthAtRhoEquals(showerStartingPosition.getRho());
  G4double pathLength = pathLength0; // this will grow along the shower development


  //--- 2.2  Fix intrinsic properties of em. showers.

  G4double fluctuatedTmax = std::log(logY - 0.7157);
  G4double fluctuatedAlpha = std::log(0.7996 +(0.4581 + 1.8628/Z)*logY);

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
  G4double averageAlpha = 0.21+(0.492+2.38/Z)*logY;
  G4double spotTmax  = averageTmax * (0.698 + .00212*Z);
  G4double spotAlpha = averageAlpha * (0.639 + .00334*Z);
  G4double spotBeta = (spotAlpha-1.0)/spotTmax;


   if(theHisto->getStoreFlag()) {
    theHisto->incE_atEcal->Fill(incomingEnergy);
    theHisto->rho_ssp->Fill(showerStartingPosition.rho());
  }

  //  parameters for lateral distribution and fluctuation
  G4double z1=0.0251+0.00319*logEinc;
  G4double z2=0.1162-0.000381*Z;

  G4double k1=0.659 - 0.00309 * Z;
  G4double k2=0.645;
  G4double k3=-2.59;
  G4double k4=0.3585+ 0.0421*logEinc;

  G4double p1=2.623 -0.00094*Z;
  G4double p2=0.401 +0.00187*Z;
  G4double p3=1.313 -0.0686*logEinc;

  //@@@ dwjang, intial tuning by comparing 20GeV TB data
  p1 = 2.47;
 
  // preparation of longitudinal integration
  G4double stepLengthLeft = fastTrack.GetEnvelopeSolid()->DistanceToOut(fastTrack.GetPrimaryTrackLocalPosition(),
									fastTrack.GetPrimaryTrackLocalDirection()) / cm;

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

  // loop for longitudinal integration
  while(energy > 0.0 && stepLengthLeft > 0.0) { 

    stepLengthLeftInX0 = stepLengthLeft / radLength;

    if ( stepLengthLeftInX0 < divisionStepInX0 ) {
      deltaZInX0 = stepLengthLeftInX0;
      deltaZ     = deltaZInX0 * radLength;
      stepLengthLeft = 0.0;
    }
    else {
      deltaZInX0 = divisionStepInX0;
      deltaZ     = deltaZInX0 * radLength;
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
    G4double emSpotEnergy = deltaEnergy / nSpotsInStep;
    GflashEnergySpot eSpot;

    if(theHisto->getStoreFlag()) {
      theHisto->dEdz->Fill(zInX0-0.5,deltaEnergy);
      theHisto->dEdz_p->Fill(zInX0-0.5,deltaEnergy);
    }

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

      G4double rShower =  rInRM * rMoliere;

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

      // put energy and position to a spot
      eSpot.setEnergy(emSpotEnergy*GeV);
      eSpot.setPosition(SpotPosition*cm);

      // for histogramming      
      G4double zInX0_spot = std::abs(pathLength+incrementPath - pathLength0)/radLength;

      if(theHisto->getStoreFlag()) {
	theHisto->rxry->Fill(rShower*std::cos(azimuthalAngle)/rMoliere,rShower*std::sin(azimuthalAngle)/rMoliere);
	theHisto->dx->Fill(rShower*std::cos(azimuthalAngle)/rMoliere);
	theHisto->xdz->Fill(zInX0-0.5,rShower*std::cos(azimuthalAngle)/rMoliere);
	theHisto->dndz_spot->Fill(zInX0_spot);
	theHisto->rzSpots->Fill(SpotPosition.z(),SpotPosition.r());
	theHisto->rArm->Fill(rShower/rMoliere);
      }

      // to be returned
      aEnergySpotList.push_back(eSpot);
      
    }
  }

}

