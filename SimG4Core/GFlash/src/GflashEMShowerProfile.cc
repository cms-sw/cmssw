//
// initial setup : Soon Jun & Dongwook Jang
// Translated from Fortran code.

#include "Randomize.hh"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/GenericFunctions/IncompleteGamma.hh"

#include "SimG4Core/GFlash/interface/GflashEMShowerProfile.h"
#include "SimG4Core/GFlash/interface/GflashHadronShowerConstants.h"
#include "SimG4Core/GFlash/interface/GflashEnergySpot.h"
#include "SimG4Core/GFlash/interface/GflashHistogram.h"
#include "FWCore/Utilities/interface/Exception.h"

GflashEMShowerProfile::GflashEMShowerProfile(G4Region* envelope)
{
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
  delete theRandGauss;
}

void GflashEMShowerProfile::parameterization(const G4FastTrack& fastTrack)
{
  // This part of code is copied from the original GFlash Fortran code.
  // Here we just ported it

  const G4double energyCutoff     = 0.01; 
  const G4int    maxNumberOfSpots = 10000;

  G4double incomingEnergy   = fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV;
  G4double radLength = 8.9;
  G4double Z = 68.360;
  G4double A = 170.871;
  G4double criticalEnergy = 2.66*std::pow(radLength*Z/A,1.1);

//   printf("GflashEMShowerProfile::parameterization, Step(%2d), incomingEnergy(%f), rho(%f), LV(%s)\n",
// 	 fastTrack.GetPrimaryTrack()->GetCurrentStepNumber(),
// 	 incomingEnergy,fastTrack.GetPrimaryTrack()->GetPosition().rho(),
// 	 fastTrack.GetPrimaryTrack()->GetTouchable()->GetVolume()->GetLogicalVolume()->GetName().c_str());

  //parameters
  G4double logEinc = std::log(incomingEnergy);
  G4double y = incomingEnergy*GeV / criticalEnergy; // y = E/Ec, criticalEnergy is in MeV
  G4double logY = std::log(y);
 
  //--- 2.2  Fix intrinsic properties of em. showers.
  G4double TM = std::log(logY -0.812);
  G4double AP = std::log(0.81 +(0.458 +2.26/Z)*logY);
  G4double STM = 1.0/( -1.4  + 1.26 * logY);
  G4double SAP = 1.0/( -0.58 + 0.86 * logY);
  G4double RHO = 0.705  - 0.023 * logY;
  G4double SQRTPL = std::sqrt((1.0+RHO)/2.0);
  G4double SQRTLE = std::sqrt((1.0-RHO)/2.0);

  G4double RNORM1 = theRandGauss->fire();
  G4double RNORM2 = theRandGauss->fire();
  G4double TMR = TM + STM*(SQRTPL*RNORM1 + SQRTLE*RNORM2);
  G4double APR = AP + SAP*(SQRTPL*RNORM1 - SQRTLE*RNORM2);

  // actual shower parameter TMAX, ALP, BET
  G4double TMAX = std::exp(TMR);
  G4double ALP = std::exp(APR);
  G4double BET = (ALP - 1.0)/TMAX;
 
  //C spot distribution parameter TSPO, ALPSPO, BETSPO
  G4double TMAV = logY-0.858;
  G4double ALAV = 0.21+(0.492+2.38/Z)*logY;
  G4double TMSPO  = TMAV * (0.698 + .00212*Z);
  G4double ALPSPO = ALAV * (0.639 + .00334*Z);
  G4double BETSPO = (ALPSPO-1.0)/TMSPO;


 
  //C 2.3  Compute rotation matrix around particle direction to convert
  //C      shower reference into detector reference:
  G4ThreeVector positionShower = fastTrack.GetPrimaryTrack()->GetPosition();
  G4ThreeVector zhatShower = fastTrack.GetPrimaryTrack()->GetMomentumDirection();
  G4ThreeVector xhatShower = zhatShower.orthogonal().unit();
  G4ThreeVector yhatShower = zhatShower.cross(xhatShower).unit();

  //end of initialization and parameterization

  if(theHisto->getStoreFlag()) {
    theHisto->incE_atEcal->Fill(incomingEnergy);
    theHisto->rho_ssp->Fill(positionShower.rho()/cm);
  }

  //C ====================================================================
  //C 3.  FETCH VOLUME SPECIFIC CONSTANTS EVERY TIME
  //C ====================================================================
  //C  ------------------------------------------------------------------
  //C 3.1 volume dependent parameters for lat. profiles.
  //C     TSPOTS=number of spots needed for lat. shape fluctuations.
  //C  ------------------------------------------------------------------


  G4double eScale = 21.2;
  G4double rMoliere = radLength * eScale / criticalEnergy;
  G4double TSPOTS = 93.0 * std::log(Z) * std::pow(incomingEnergy,0.876);

  G4double RC1=0.0251+0.00319*logEinc; //z1
  //  G4double RC2=0.1162-0.000381*Z;  //z2
  G4double RC2=2.0*0.1162-0.000381*Z;  //z2

  G4double RT1=0.659 - 0.00309 * Z; //k1
  G4double RT2=0.645; //k2
  G4double RT3=-2.59; //k3 
  G4double RT4=0.3585+ 0.0421*logEinc; //k4

  G4double RP1=2.623 -0.00094*Z;  //p1
  G4double RP2=0.401 +0.00187*Z; //p2
  G4double RP3=1.313 -0.0686*logEinc; //p3
 
  //C 4. LONGITUDINAL INTEGRATION OF THE SHOWER (REPEAT-UNTIL LOOP)

  //syjun-we should consider magnetic field - i.e., helix instead of a straight line
  G4double stepLengthLeft = fastTrack.GetEnvelopeSolid()->
      DistanceToOut(fastTrack.GetPrimaryTrackLocalPosition(),
                    fastTrack.GetPrimaryTrackLocalDirection());

  G4double zInX0 = 0.0;
  G4double deltaZInX0 = 0.0;
  G4double deltaZ = 0.0;
  G4double stepLengthLeftInX0 = 0.0;

  const G4double divisionStepInX0 = 1.0; //step size in the radiation lenth
  G4double energy = incomingEnergy;

  Genfun::IncompleteGamma gammaDist;
  
  G4double ARTIMS = 0.0;
  G4double EGAM = 0.0;
  G4double EGOLD = 0.0;
  G4double SGOLD = 0.0;
  G4double SGAM  = 0.;

  //energy segment in Gamma distribution of shower in each step  
  G4double deltaEnergy =0.0 ;
  G4int indexSpot = 0;

  //step increment along the shower direction
  G4double deltaStep = 0.0;

  // for debugging
  G4double energyDeposited = 0.0;

  while(energy > 0.0 && stepLengthLeft > 0.0) { 

  //C 5.1   Find integration width and shower depth in X0 for this step

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


    G4int numSpotInStep = 0;

    if ( energy > energyCutoff  ) {
      EGOLD  = EGAM;
      gammaDist.a().setValue(ALP);  //alpha
      EGAM = gammaDist(BET*zInX0); //beta
      G4double EGACT  = EGAM - EGOLD;
      deltaEnergy   = std::min(energy,incomingEnergy*EGACT);
 
      SGOLD  = SGAM;
      gammaDist.a().setValue(ALPSPO);  //alpha spot
      SGAM = gammaDist(BETSPO*zInX0); //beta spot
      numSpotInStep = std::max(1,int(TSPOTS * (SGAM - SGOLD)));
    }
    else {
      deltaEnergy = energy;
      SGOLD  = SGAM;
      numSpotInStep = std::max(1,int(TSPOTS * (1.0 - SGOLD)));
    }

    if ( deltaEnergy > energy || (energy-deltaEnergy) < energyCutoff ) deltaEnergy = energy;

    energy  -= deltaEnergy;

    if ( indexSpot+numSpotInStep > maxNumberOfSpots ) {
      numSpotInStep = maxNumberOfSpots - indexSpot;
      if ( numSpotInStep < 1 ) { // @@ check
        std::cout << "GflashEMShowerProfile::Parameterization : Too Many Spots " << std::endl;
        std::cout << "                       break to regenerate numSpotInStep " << std::endl;
        break;
      }
    }


    //C 5.3  Linear transport in direction of incident particle

    deltaStep      += 0.5*deltaZ;
    positionShower += deltaStep*zhatShower;
    deltaStep      =  0.5*deltaZ;


    //C 5.5  lateral shape and fluctuations for  homogenous calo.

    G4double TSCALE = TMAX *ALP/(ALP-1.0) * (std::exp(AP)-1.0)/std::exp(AP);
    G4double R50 = std::min(10.0,(zInX0 - 0.5*deltaZInX0)/TSCALE);
    G4double RCORE = RC1 + RC2 * R50; 
    G4double RTAIL = RT1 *( std::exp(RT3*(R50-RT2)) + std::exp(RT4*(R50-RT2))); // @@ check RT3 sign
    G4double RPDUM = (RP2 - R50)/RP3;
    G4double RPROP = RP1 *  std::exp( RPDUM - std::exp(RPDUM) );


    ARTIMS += rMoliere * deltaZInX0;
    G4double AVALAT = ARTIMS / zInX0;

    //C 5.6  Deposition of spots according to lateral distr.

    G4double emSpotEnergy = deltaEnergy / numSpotInStep;
    GflashEnergySpot eSpot;

    // for debugging
    energyDeposited += deltaEnergy;

    if(theHisto->getStoreFlag()) {
      theHisto->dEdz->Fill(zInX0-0.5,deltaEnergy);
      theHisto->dEdz_p->Fill(zInX0-0.5,deltaEnergy);
    }

    for (G4int ispot = 0 ;  ispot < numSpotInStep ; ispot++) {

      indexSpot++;

      G4double UNI1 = G4UniformRand();
      G4double UNI2 = G4UniformRand();
      G4double RINRM = 0.0;

      if (UNI1 < RPROP) {
	RINRM = RCORE * std::sqrt( UNI2/(1.0-UNI2) );
      }
      else {
	RINRM = RTAIL * std::sqrt( UNI2/(1.0-UNI2) );
      }

      G4double rShower =  RINRM * AVALAT;

      //c ---  uniform smearing in phi
      G4double azimuthalAngle = twopi*G4UniformRand();

      // --- Compute space point in detector reference
      G4ThreeVector SpotPosition = positionShower +
	rShower*std::cos(azimuthalAngle)*xhatShower +
	rShower*std::sin(azimuthalAngle)*yhatShower +
	(deltaZ/numSpotInStep)*(ispot+0.5 - 0.5*numSpotInStep)*zhatShower;

      eSpot.setEnergy(emSpotEnergy*GeV);
      eSpot.setPosition(SpotPosition);
      
      float zInX0_spot = std::abs((SpotPosition-fastTrack.GetPrimaryTrack()->GetPosition()).dot(zhatShower))/radLength;
      float signedR = std::cos(azimuthalAngle) > 0.0 ? SpotPosition.r() : -SpotPosition.r();

      if(theHisto->getStoreFlag()) {
	theHisto->rxry->Fill(rShower*std::cos(azimuthalAngle)/rMoliere,rShower*std::sin(azimuthalAngle)/rMoliere);
	theHisto->dx->Fill(rShower*std::cos(azimuthalAngle)/rMoliere);
	theHisto->xdz->Fill(zInX0-0.5,rShower*std::cos(azimuthalAngle)/rMoliere);
	theHisto->dndz_spot->Fill(zInX0_spot);
	theHisto->rzSpots->Fill(SpotPosition.z()/cm,signedR/cm);
      }

      aEnergySpotList.push_back(eSpot);

      
    }
  }

}

