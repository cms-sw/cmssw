//
// $Id: GflashEMShowerProfile.cc,v 1.5 2010/04/30 19:10:11 dwjang Exp $
// initial setup : Soon Jun & Dongwook Jang
// Translated from Fortran code.
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimGeneral/GFlash/interface/GflashEMShowerProfile.h"
#include "SimGeneral/GFlash/interface/GflashTrajectory.h"
#include "SimGeneral/GFlash/interface/GflashTrajectoryPoint.h"
#include "SimGeneral/GFlash/interface/GflashHit.h"
#include "SimGeneral/GFlash/interface/Gflash3Vector.h"
#include "SimGeneral/GFlash/interface/GflashHistogram.h"
#include "SimGeneral/GFlash/interface/GflashShowino.h"

#include <CLHEP/GenericFunctions/IncompleteGamma.hh>
#include <CLHEP/Units/PhysicalConstants.h>
#include <CLHEP/Random/Randomize.h>
#include <CLHEP/Random/RandGaussQ.h>

GflashEMShowerProfile::GflashEMShowerProfile(edm::ParameterSet parSet) : theParSet(parSet)
{
  theBField = parSet.getParameter<double>("bField");
  theEnergyScaleEB = parSet.getParameter<double>("energyScaleEB");
  theEnergyScaleEE = parSet.getParameter<double>("energyScaleEE");

  jCalorimeter = Gflash::kNULL;

  theShowino = new GflashShowino();

  theHisto = GflashHistogram::instance();

}


GflashEMShowerProfile::~GflashEMShowerProfile()
{
  if(theShowino) delete theShowino;
}

void GflashEMShowerProfile::initialize(int showerType, double energy, double globalTime, double charge,
				       Gflash3Vector &position,Gflash3Vector &momentum) 
{
  theShowino->initialize(showerType, energy, globalTime, charge,
                         position, momentum, theBField);
}

void GflashEMShowerProfile::parameterization()
{
  // This part of code is copied from the original GFlash Fortran code.
  // reference : hep-ex/0001020v1
  // The units used in Geant4 internally are in mm, MeV.
  // For simplicity, the units here are in cm, GeV.

  const double energyCutoff     = 0.01; 
  const int    maxNumberOfSpots = 100000;

  double incomingEnergy   = theShowino->getEnergy();
  Gflash3Vector showerStartingPosition = theShowino->getPositionAtShower();

  //find the calorimeter at the shower starting point
  jCalorimeter = Gflash::getCalorimeterNumber(showerStartingPosition);

  double logEinc = std::log(incomingEnergy);
  double y = incomingEnergy / Gflash::criticalEnergy; // y = E/Ec, criticalEnergy is in GeV
  double logY = std::log(y);

  // Total number of spots are not yet optimized.
  double nSpots = 93.0 * std::log(Gflash::Z[jCalorimeter]) * std::pow(incomingEnergy,0.876);

  //path Length from the origin to the shower starting point in cm
  double pathLength0 = theShowino->getPathLengthAtShower();
  double pathLength = pathLength0; // this will grow along the shower development

  //--- 2.2  Fix intrinsic properties of em. showers.

  double fluctuatedTmax = std::log(logY - 0.7157);
  double fluctuatedAlpha = std::log(0.7996 +(0.4581 + 1.8628/Gflash::Z[jCalorimeter])*logY);

  double sigmaTmax = 1.0/( -1.4  + 1.26 * logY);
  double sigmaAlpha = 1.0/( -0.58 + 0.86 * logY);
  double rho = 0.705  - 0.023 * logY;
  double sqrtPL = std::sqrt((1.0+rho)/2.0);
  double sqrtLE = std::sqrt((1.0-rho)/2.0);

  double norm1 =  CLHEP::RandGaussQ::shoot();
  double norm2 =  CLHEP::RandGaussQ::shoot();
  double tempTmax = fluctuatedTmax + sigmaTmax*(sqrtPL*norm1 + sqrtLE*norm2);
  double tempAlpha = fluctuatedAlpha + sigmaAlpha*(sqrtPL*norm1 - sqrtLE*norm2);

  // tmax, alpha, beta : parameters of gamma distribution
  double tmax = std::exp(tempTmax);
  double alpha = std::exp(tempAlpha);
  double beta = std::max(0.0,(alpha - 1.0)/tmax);
 
  // spot fluctuations are added to tmax, alpha, beta
  double averageTmax = logY-0.858;
  double averageAlpha = 0.21+(0.492+2.38/Gflash::Z[jCalorimeter])*logY;
  double spotTmax  = averageTmax * (0.698 + .00212*Gflash::Z[jCalorimeter]);
  double spotAlpha = averageAlpha * (0.639 + .00334*Gflash::Z[jCalorimeter]);
  double spotBeta = std::max(0.0,(spotAlpha-1.0)/spotTmax);

   if(theHisto->getStoreFlag()) {
    theHisto->em_incE->Fill(incomingEnergy);
    theHisto->em_ssp_rho->Fill(showerStartingPosition.rho());
    theHisto->em_ssp_z->Fill(showerStartingPosition.z());
  }

  //  parameters for lateral distribution and fluctuation
  double z1=0.0251+0.00319*logEinc;
  double z2=0.1162-0.000381*Gflash::Z[jCalorimeter];

  double k1=0.659 - 0.00309 * Gflash::Z[jCalorimeter];
  double k2=0.645;
  double k3=-2.59;
  double k4=0.3585+ 0.0421*logEinc;

  double p1=2.623 -0.00094*Gflash::Z[jCalorimeter];
  double p2=0.401 +0.00187*Gflash::Z[jCalorimeter];
  double p3=1.313 -0.0686*logEinc;

  // @@@ dwjang, intial tuning by comparing 20-150GeV TB data : e25Scale = 1.006 for EB with ecalNotContainment = 1.0.
  // Now e25Scale is a configurable parameter with default ecalNotContainment which is 0.97 for EB and 0.975 for EE.
  // So if ecalNotContainment constants are to be changed in the future, e25Scale should be changed accordingly.
  double e25Scale = 1.0;
  if(jCalorimeter == Gflash::kESPM) e25Scale = theEnergyScaleEB;
  else if(jCalorimeter == Gflash::kENCA) e25Scale = theEnergyScaleEE;

  // @@@ dwjang, intial tuning by comparing 20-150GeV TB data : p1 *= 0.965
  p1 *= 0.965;

  // preparation of longitudinal integration
  double stepLengthLeft = getDistanceToOut(jCalorimeter);

  int    nSpots_sd = 0; // count total number of spots in SD
  double zInX0 = 0.0; // shower depth in X0 unit
  double deltaZInX0 = 0.0; // segment of depth in X0 unit
  double deltaZ = 0.0; // segment of depth in cm
  double stepLengthLeftInX0 = 0.0; // step length left in X0 unit

  const double divisionStepInX0 = 0.1; //step size in X0 unit
  double energy = incomingEnergy; // energy left in GeV

  Genfun::IncompleteGamma gammaDist;

  double energyInGamma = 0.0; // energy in a specific depth(z) according to Gamma distribution
  double preEnergyInGamma = 0.0; // energy calculated in a previous depth
  double sigmaInGamma  = 0.; // sigma of energy in a specific depth(z) according to Gamma distribution
  double preSigmaInGamma = 0.0; // sigma of energy in a previous depth

  //energy segment in Gamma distribution of shower in each step  
  double deltaEnergy =0.0 ; // energy in deltaZ
  int spotCounter = 0; // keep track of number of spots generated

  //step increment along the shower direction
  double deltaStep = 0.0;

  theGflashHitList.clear();

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

    int nSpotsInStep = 0;

    if ( energy > energyCutoff  ) {
      preEnergyInGamma  = energyInGamma;
      gammaDist.a().setValue(alpha);  //alpha
      energyInGamma = gammaDist(beta*zInX0); //beta
      double energyInDeltaZ  = energyInGamma - preEnergyInGamma;
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
	edm::LogInfo("SimGeneralGFlash") << "GflashEMShowerProfile: Too Many Spots ";
        edm::LogInfo("SimGeneralGFlash") << " - break to regenerate nSpotsInStep ";
        break;
      }
    }

    // It begins with 0.5 of deltaZ and then icreases by 1 deltaZ
    deltaStep  += 0.5*deltaZ;
    pathLength += deltaStep;
    deltaStep   =  0.5*deltaZ;

    // lateral shape and fluctuations for  homogenous calo.
    double tScale = tmax *alpha/(alpha-1.0) * (std::exp(fluctuatedAlpha)-1.0)/std::exp(fluctuatedAlpha);
    double tau = std::min(10.0,(zInX0 - 0.5*deltaZInX0)/tScale);
    double rCore = z1 + z2 * tau; 
    double rTail = k1 *( std::exp(k3*(tau-k2)) + std::exp(k4*(tau-k2))); // @@ check RT3 sign
    double p23 = (p2 - tau)/p3;
    double probabilityWeight = p1 *  std::exp( p23 - std::exp(p23) );

    // Deposition of spots according to lateral distr.
    // Apply absolute energy scale
    // Convert into MeV unit
    double hitEnergy = deltaEnergy / nSpotsInStep * e25Scale * CLHEP::GeV;
    double hitTime = theShowino->getGlobalTime()*CLHEP::nanosecond + (pathLength - pathLength0)/30.0;

    GflashHit aHit;

    for (int ispot = 0 ;  ispot < nSpotsInStep ; ispot++) {
      spotCounter++;

      // Compute global position of generated spots with taking into account magnetic field
      // Divide deltaZ into nSpotsInStep and give a spot a global position
      double incrementPath = (deltaZ/nSpotsInStep)*(ispot+0.5 - 0.5*nSpotsInStep);

      // trajectoryPoint give a spot an imaginary point along the shower development
      GflashTrajectoryPoint trajectoryPoint;
      theShowino->getHelix()->getGflashTrajectoryPoint(trajectoryPoint,pathLength+incrementPath);

      double rShower = 0.0;
      Gflash3Vector hitPosition = locateHitPosition(trajectoryPoint,rCore,rTail,probabilityWeight,rShower);

      // Convert into mm unit
      hitPosition *= CLHEP::cm;

      if( std::fabs(hitPosition.getZ()/CLHEP::cm) > Gflash::Zmax[Gflash::kHE]) continue;

      // put energy and position to a Hit
      aHit.setTime(hitTime);
      aHit.setEnergy(hitEnergy);
      aHit.setPosition(hitPosition);
      theGflashHitList.push_back(aHit);

      double zInX0_spot = std::abs(pathLength+incrementPath - pathLength0)/Gflash::radLength[jCalorimeter];

      nSpots_sd++;

      // for histogramming      
      if(theHisto->getStoreFlag()) {
	theHisto->em_long->Fill(zInX0_spot,hitEnergy/CLHEP::GeV);
	theHisto->em_lateral->Fill(zInX0_spot,rShower/Gflash::rMoliere[jCalorimeter],hitEnergy/CLHEP::GeV);
	theHisto->em_long_sd->Fill(zInX0_spot,hitEnergy/CLHEP::GeV);
	theHisto->em_lateral_sd->Fill(zInX0_spot,rShower/Gflash::rMoliere[jCalorimeter],hitEnergy/CLHEP::GeV);
      }
      
    } // end of for spot iteration

  } // end of while for longitudinal integration

  if(theHisto->getStoreFlag()) {
    theHisto->em_nSpots_sd->Fill(nSpots_sd);
  }

  //  delete theGflashNavigator;

}

double GflashEMShowerProfile::getDistanceToOut(Gflash::CalorimeterNumber kCalor) {

  double stepLengthLeft = 0.0;
  if(kCalor == Gflash::kESPM ) {
    stepLengthLeft = theShowino->getHelix()->getPathLengthAtRhoEquals(Gflash::Rmax[Gflash::kESPM])
                   - theShowino->getPathLengthAtShower();
  }
  else if (kCalor == Gflash::kENCA) {
    double zsign = (theShowino->getPosition()).getEta() > 0 ? 1.0 : -1.0;
    stepLengthLeft = theShowino->getHelix()->getPathLengthAtZ(zsign*Gflash::Zmax[Gflash::kENCA])
                   - theShowino->getPathLengthAtShower();
  }
  return stepLengthLeft;

}

Gflash3Vector GflashEMShowerProfile::locateHitPosition(GflashTrajectoryPoint& point, 
				     double rCore, double rTail, double probability,double &rShower)
{
  double u1 = CLHEP::HepUniformRand();
  double u2 = CLHEP::HepUniformRand();
  double rInRM = 0.0;
  
  if (u1 < probability ) {
    rInRM = rCore* std::sqrt( u2/(1.0-u2) );
  }
  else {
    rInRM = rTail * std::sqrt( u2/(1.0-u2) );
  }
  
  rShower =  rInRM * Gflash::rMoliere[jCalorimeter];

  // Uniform & random rotation of spot along the azimuthal angle
  double azimuthalAngle = CLHEP::twopi*CLHEP::HepUniformRand();

  // actual spot position by adding a radial vector to a trajectoryPoint
  Gflash3Vector position = point.getPosition() +
    rShower*std::cos(azimuthalAngle)*point.getOrthogonalUnitVector() +
    rShower*std::sin(azimuthalAngle)*point.getCrossUnitVector();
  
  return position;
}
