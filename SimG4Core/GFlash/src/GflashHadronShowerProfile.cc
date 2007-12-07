#include "SimG4Core/GFlash/interface/GflashHadronShowerProfile.h"
//#include "SimG4Core/GFlash/interface/GflashHadronShowerConstants.h"
#include "SimG4Core/GFlash/interface/GflashEnergySpot.h"
//#include "SimG4Core/GFlash/interface/GflashMediaMap.h"
#include "SimG4Core/GFlash/interface/GflashHistogram.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/GenericFunctions/IncompleteGamma.hh"
#include "Randomize.hh"

GflashHadronShowerProfile::GflashHadronShowerProfile(G4Region* envelope)
{
  jCalorimeter = Gflash::kNULL;
  showerType   = 0;

  fillFluctuationVector();

  //  theGflashStep = new G4Step();

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
  theRandGamma = new CLHEP::RandGamma(rng->getEngine());

}

GflashHadronShowerProfile::~GflashHadronShowerProfile()
{
  //  delete theGflashStep;
  delete theRandGauss;
  delete theRandGamma;
}

Gflash::CalorimeterNumber GflashHadronShowerProfile::getCalorimeterNumber(const G4FastTrack& fastTrack)
{
  G4String logicalVolumeName = fastTrack.GetEnvelopeLogicalVolume()->GetName();
  //@@@@@@@@@@@@@@@@@@
  //  GflashCalorimeterNumber index = theMediaMap->getIndex(logicalVolumeName);
  Gflash::CalorimeterNumber index = Gflash::kESPM;
  return index;
}

void GflashHadronShowerProfile::hadronicParameterization(const G4FastTrack& fastTrack)
{

  // This methods is based on the fortran code GFSHOW originally written by 
  // S. Peters and G. Grindhammer (also see NIM A290 (1990) 469-488)
  //

  //unit convention: energy in [GeV] and length in [cm]

  //number of shower types

  //  const G4int MIPC = 3;

  // maximum number of energy spots = 80.*std::log(einc)+50
  const G4int    maxNumberOfSpots = 1000;  

  // low energy cutoff (unit in GeV)
  const G4double energyCutoff     = 0.01; 

  // intrinsic properties of hadronic showers (lateral shower profile)
  const G4double radialScale          = 0.9; 
  const G4double maxShowerDepthforR50 = 2.0;
  const G4double maxR50               = 1.4;
  const G4double maxRShower           = 2.6;

  // 2.1  initialization of GFLASH parameters: - once at the shower starting point

  // zConv: conversion from absl to radl:

  //  GflashCalorimeterNumber jCalorimeter = theMediaMap->getCalorimeterNumber(fastTrack);

  //@@@@@@@@@@@@@@@@@@@@@
  //  jCalorimeter = theMediaMap->getCalorimeterNumber(fastTrack);
  jCalorimeter = Gflash::kESPM;

  // PbWO4 : absLength = 20.7394 cm, radLength = 0.892414 cm
  // Brass : absLength = 16.3898 cm, radLength = 1.502860 cm

  // @@@@@@@@@@@@@
  //  G4double absLength = theMediaMap->getMaterial(jCalorimeter)->GetNuclearInterLength()/cm;
  //  G4double radLength = theMediaMap->getMaterial(jCalorimeter)->GetRadlen()/cm; 

  G4double absLength = 20.7394;
  G4double radLength = 0.892414; 

  G4double zConv[4] = {absLength/radLength,1.,absLength/radLength,1.};

  G4double deltaStep = 0.;
  
  G4double varianceLateral = 0.;
  G4double rShower = 0.;
  G4double rGauss = theRandGauss->fire();

  // 2.2  Compute rotation matrix around particle direction 
  // to convert shower reference into detector reference:

  // axis of the shower, in global reference frame

  G4ThreeVector zhatShower = fastTrack.GetPrimaryTrack()->GetMomentumDirection();
  G4ThreeVector xhatShower = zhatShower.orthogonal().unit();
  G4ThreeVector yhatShower = zhatShower.cross(xhatShower).unit();

  //fix intrinsic properties of hadronic showers - i.e., set parameters

  G4double einc   = fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV;

  //get all necessary parameters for hadronic shower profiles
  loadParameters(einc);

  G4double energy = energyToDeposit;

  G4double energyByShowerType[Gflash::MIPC] = {0.};
  for (G4int i = 0; i < showerType ; i++) {
    energyByShowerType[i] = frac[i] * energyToDeposit * (1.- frac[3]);
  }

  G4double zInLx[Gflash::MIPC]  = {0.};

  //conv. length for lateral spreading of had sh.

  // 3.2  Spot energy to simulate sampling fluctuations (SampleEnergySpot) and
  //     number of spots needed to fullfill geometry constraints(TotalNumberOfSpots):

  G4int numberOfSpots = std::max( 50, static_cast<int>(80.*std::log(einc)+50.));

  // Limit number of spots to maxNumberOfSpots

  numberOfSpots = std::min(numberOfSpots,maxNumberOfSpots);
  G4double  spotEnergy = energyToDeposit/numberOfSpots;

  const G4double divisionStepInL0 = 0.1; //step size in the absorption lenth

  G4double zInL0 = 0.0;
  G4double deltaZInL0 = 0.0;
  G4double stepLengthLeftInL0 = 0.0;
  G4double deltaZ;

  //@@@ deposit energy below shower starting point only at the beginning
  //@@@ of a hadronic shower
  //@@@ deltaEnergy   = energyToDeposit * frac[3];
  //@@@ stepLength  = stepLength - alpha[3]/beta[3]*absLength;
  //@@@ hadronicFraction = 1.

  //@@@syjun-we should consider magnetic field - i.e., helix instead of a straight line
  //@@@syjun-also if the shower start at EM, this should be wrong

  G4double stepLengthLeft = fastTrack.GetEnvelopeSolid()->
    DistanceToOut(fastTrack.GetPrimaryTrackLocalPosition(),
		  fastTrack.GetPrimaryTrackLocalDirection())/cm;

  //temporarily add 1m for hadronic calorimeter - distance to the end of HCAL
  if(jCalorimeter == Gflash::kESPM || jCalorimeter == Gflash::kENCA ) stepLengthLeft += 100.0;

  //Incomplete Gamma function gives an intergrate probability of the longitudinal shower
  //profile to the shower depth (z): 
  //Energy in a delta step (dz) = (energy to deposite)*[Gamma(z+dz)-Gamma(z)]*dz

  Genfun::IncompleteGamma gammaDist;

  G4int totalNumberOfSpots = 0;

  // The shower starting point (PostStepPoint of Hadronic Inelestic interaction)
  // is the position of the track after the step of the current process is updated
                                                                                                   
  G4ThreeVector positionShower = fastTrack.GetPrimaryTrack()->GetPosition();

  //@@@debug histograms

  //empty energy spot vector for a new track
  aEnergySpotList.clear();

  //debug 
  int counterLoop = 0;

  G4double pgam = 0.;
  G4double sgam[Gflash::MIPC] = {0.};

  while(energy > 0.0 && stepLengthLeft > 0.0) {

    counterLoop++;      

    // Find integration width and shower depth in L0 for this step

    stepLengthLeftInL0 = stepLengthLeft / absLength;
    
    if ( stepLengthLeftInL0 < divisionStepInL0 ) {
      deltaZInL0 = stepLengthLeftInL0;
      deltaZ     = deltaZInL0 * absLength;
      stepLengthLeft  = 0.0;
    }
    else {
      deltaZInL0 = divisionStepInL0;
      deltaZ     = deltaZInL0 * absLength;
      stepLengthLeft -= deltaZ;
    }

    zInL0 =+ deltaZInL0;

    // 5.2  Integration of longitudinal shower profile

    double deltaEnergy = 0.;
    double dE[Gflash::MIPC] = {0.};

    /*
    for(G4int i = 0; i < showerType ; i++) {
      if( energyByShowerType[i] > energyCutoff ) {
        zInLx[i] += deltaZInL0*zConv[i];
	gammaDist.a().setValue(alpha[i]);
	dE[i]    = energyToDeposit*frac[i]*gammaDist(beta[i]*zInLx[i]);
	dE[i]    = std::min(energyByShowerType[i],dE[i]);
	energyByShowerType[i] -= dE[i];
      }
      else {
        dE[i] = energyByShowerType[i];
	energyByShowerType[i] = 0.;
      }
      deltaEnergy += dE[i];
    }
    */

    for(G4int i = 0; i < showerType ; i++) {

      if( energyByShowerType[i] > energyCutoff ) {
        zInLx[i] += deltaZInL0*zConv[i];
	gammaDist.a().setValue(alpha[i]);
        pgam     = sgam[i];
        sgam[i]  = gammaDist(beta[i]*zInLx[i]);
	dE[i]    = energyToDeposit*frac[i]*(gammaDist(beta[i]*zInLx[i])-pgam);
	dE[i]    = std::min(energyByShowerType[i],dE[i]);
	energyByShowerType[i] -= dE[i];
      }
      else {
        dE[i] = energyByShowerType[i];
	energyByShowerType[i] = 0.;
      }
      deltaEnergy += dE[i];
    }


    if((deltaEnergy > energy ) || ((energy - deltaEnergy) < energyCutoff )) {
      deltaEnergy = energy;
    }
    energy -= deltaEnergy;

    //@@@debug
    /*
    std::cout << "energy to Deposit : countLoop : delta Energy : Energy Left " << energyToDeposit << " " << counterLoop << " "
	      << deltaEnergy << " " << energy << std::endl;
    */
    G4double hadronicFraction = (deltaEnergy == 0.0) ? 0.0 : dE[0]/deltaEnergy;

    deltaStep      += 0.5*deltaZ;
    positionShower += deltaStep*zhatShower;
    deltaStep      =  0.5*deltaZ;
    
    // Sampling fluctuations determine the number of spots:
    G4double fluctuatedEnergy = deltaEnergy;
    
    // sampling fluctuation only for HCAL

    if (insideSampling(positionShower)) samplingFluctuation(fluctuatedEnergy,einc); 

    //

    G4int numSpotInStep = std::max(1,static_cast<int>(deltaEnergy/spotEnergy));
    
    if ( totalNumberOfSpots+numSpotInStep > maxNumberOfSpots ) {
      numSpotInStep = maxNumberOfSpots - totalNumberOfSpots;
      if ( numSpotInStep <= 1 ) {
        std::cout << "GflashHadronShowerProfile::Parameterization : Too Many Spots " << std::endl;
        std::cout << "                           break to regenerate numSpotInStep " << std::endl;
        break;        
      }
    }
    
    G4double sampleSpotEnergy = fluctuatedEnergy/numSpotInStep;
    G4double hadSpotEnergy = std::max(0.,sampleSpotEnergy * hadronicFraction);
    G4double emSpotEnergy  = std::max(0.,sampleSpotEnergy - hadSpotEnergy);

    hadSpotEnergy *= Gflash::PBYMIP[jCalorimeter];

    //@@@ I don't think that these are necessary - check later
    /* 
       EDEPSP = deltaEnergy/FLOAT(numSpotInStep)
       EINDEP = EDEPSP * FINV
       EHADEP = EDEPSP * hadronicFraction
       EEMDEP = EDEPSP - EHADEP
    */
    
    // 5.5  Lateral shape and fluctuations
    // ---       R50 in absorbtion lengths rsp. rmoliere
    
    double showerDepth = std::min((zInL0-deltaZInL0), maxShowerDepthforR50);

    double R50          = lateralPar[0] + lateralPar[1] * showerDepth;
    double varinanceR50 = std::pow((lateralPar[2] + lateralPar[3] * showerDepth) * R50, 2);

    // Simulation of lognormal distribution

    double sigmaSq = std::log(varinanceR50/(R50*R50)+1.0);
    double sigmaR50 = std::sqrt(sigmaSq);
    double meanR50  = std::log(R50) - (sigmaSq/2.);

    R50    = std::min(maxR50,std::exp(rGauss*sigmaR50 + meanR50));

    // ---    Averaging lat. scale and converting  to cm

    const G4double rLatToCm = Gflash::RLTHAD[jCalorimeter];
    varianceLateral =+ rLatToCm*deltaZInL0/zInL0;
    R50    =  R50    * varianceLateral;

    double rxMax  = maxRShower * varianceLateral;
    
    //  
    GflashEnergySpot eSpot;

    for (G4int ispot = 0 ;  ispot < numSpotInStep ; ispot++) {

      totalNumberOfSpots++;

      // ---  Smearing in r according to f(r)= 2.*r*R50**2/(r**2+R50**2)**2
      G4double rnunif = G4UniformRand()*radialScale;
      G4double rxPDF  = std::sqrt(rnunif/(1.-rnunif));
      rShower  = std::min( R50 * rxPDF, rxMax);

      // ---  Uniform smearing in phi, for 66% of lateral containm.
      G4double azimuthalAngle = 0.0; 

      //      if ( rnunif < .66 ) azimuthalAngle = twopi*G4UniformRand(); 
      azimuthalAngle = twopi*G4UniformRand(); 

      // --- Compute space point in detector reference
      G4ThreeVector SpotPosition = positionShower + 
	xhatShower*rShower*std::cos(azimuthalAngle) + 
	yhatShower*rShower*std::sin(azimuthalAngle) +
	deltaZ/static_cast<float>(numSpotInStep)*zhatShower*(ispot+1/2.-numSpotInStep/2.);

      if(theHisto->getStoreFlag()) {
	theHisto->rshower->Fill(rShower);
	theHisto->lateralx->Fill(rShower*std::cos(azimuthalAngle));
	theHisto->lateraly->Fill(rShower*std::sin(azimuthalAngle));
      }      
      //convert unit of energy to geant4 default MeV
      eSpot.setEnergy((hadSpotEnergy+emSpotEnergy)*GeV);
      eSpot.setPosition(SpotPosition);
      aEnergySpotList.push_back(eSpot);

    }
  }

}

void GflashHadronShowerProfile::loadParameters(G4double einc)
{

  // C++ version of gfinha
  // Initialization of longitudinal and lateral parameters for 
  // hadronic showers. Simulation of the intrinsic fluctuations

  G4double logEinc = std::log(einc);

  G4double X[Gflash::NXN];

  // hadronic shower type according to different pi0 contributions
  // showerType = 1 : no pi0 contribution (pure hadronic)
  //            = 2 : pi0 contribution only from 1st inelastic interaction
  //            = 3 : pi0 contribution also from further interactions

  showerType = 1;
  G4int matrixPosition = Gflash::NSTRTR[0];

  if ( einc > 3.0 ) matrixPosition = Gflash::NSTRTR[1];

  G4double profOfPi0 = G4UniformRand();
  G4double probOfFirstPi0 = std::tanh(.241 * (einc + .08));
  G4double probOfLatePi0 = .5 + .5*std::tanh(1.11 * (logEinc - 2.10));

  if ( profOfPi0 < probOfFirstPi0 ) {

    showerType = 2;
    matrixPosition = Gflash::NSTRTR[2];

    if ( einc > 5.0 ) matrixPosition = Gflash::NSTRTR[3];
    if ( profOfPi0 < probOfLatePi0 ) {
 
      showerType = 3;
      matrixPosition = Gflash::NSTRTR[4];
 
      if ( einc > 22.4) matrixPosition = Gflash::NSTRTR[5];
      if ( einc > 70.7) matrixPosition = Gflash::NSTRTR[6];
      if ( einc > 141.) matrixPosition = Gflash::NSTRTR[7];
    }
  }

  G4int dimX = 3*showerType;

  //  GflashCalorimeterNumber jcalor = getCalorimeterNumber(); 
  Gflash::CalorimeterNumber jcalor = Gflash::kESPM; 

  G4double fracEnergyDeposit = 0.6851+0.1605*std::tanh( 1.0557*(logEinc-1.7654));
  G4double meanPi0Fraction   = 0.5229-0.4219*std::tanh(-0.7860*(logEinc-1.3790));

  G4double aMean[Gflash::NXN];
  G4double aSigm[Gflash::NXN];

  aMean[0] = fracEnergyDeposit;
  aSigm[0] = aMean[0] *(Gflash::FLUHAD[0][jcalor]/std::sqrt(einc)
		      + Gflash::FLUHAD[1][jcalor]/einc + Gflash::FLUHAD[2][jcalor] );
  aMean[3] =  std::max(0.1,std::log(-1.59+2.249*std::pow(std::max(logEinc,.6),0.55)));
  aSigm[3] =  std::min( 2., .385 + 1.58*std::exp(-.863*logEinc) );
  aMean[6] =  -1.212;
  aSigm[6] =  std::min( 2., .427 + 1.48*std::exp(-.858*logEinc) );

  if ( showerType >= 2 ) {

    aMean[1] = meanPi0Fraction / probOfFirstPi0;
    aSigm[1] = std::max(.1, .224 - .014*logEinc );
    aMean[4] = std::log(std::max(0.95,4.454 + 1.42*logEinc));
    aSigm[4] = std::max(.2 , .735 - .077*logEinc );
    aMean[7] = std::log(.3312 );
    aSigm[7] = std::min(2., .191 + 2.70*std::exp(-.952*logEinc));

    if ( showerType >= 3 ) {
      aMean[2] = std::min(1.,(0.325+.069*std::tanh(3.814*(logEinc-4.1)))/probOfLatePi0);
      aSigm[2] = 0.316;
      aMean[4] = std::log(std::max(1.,2.335 + 1.42*logEinc ));
      aMean[5] = std::log(2.2145);
      aSigm[5] = std::min(1.5, .424 + 2.69*std::exp(-.856*logEinc));
      aMean[8] = std::max(0.6,2.03 - .335*logEinc);
      aSigm[8] = std::max(0.4,std::min(2., 2.05 - .264*logEinc ));
    }
  }

  double normalZ[Gflash::NXN];
  for (int i = 0; i < dimX; i++) normalZ[i] = theRandGauss->fire(); 

  for (int i = 0; i < dimX; i++) {
    double Sum = 0.;
    for(int j=0; j < i ; j++) 
      Sum += RMX[matrixPosition+i*(i+1)/2+j]*aSigm[Gflash::IT[showerType-1][i]-1]*normalZ[j];
    X[i] =  Sum + aMean[Gflash::IT[showerType-1][i]-1];
  }

  G4double  fracEnergyByFirstPi0 = 0.0;
  G4double  fracEnergyByLatePi0 = 0.0;

  fracEnergyDeposit  = X[0];

  if (showerType == 2) {
    fracEnergyByFirstPi0 = X[1];
  }
  else if (showerType == 3) {
    fracEnergyByFirstPi0 = X[1];
    fracEnergyByLatePi0 = X[2];
  }

  for(G4int i = 0 ; i < showerType ; i++) {
    beta [i] = std::exp(X[2*showerType+i]);
    alpha[i] = std::exp(X[showerType+i])*beta[i];
  }

  //fluctuate energy deposited in front of shower starting point 

  G4double febss;
  febss = (0.471 * std::exp(-.958 * std::log(einc+1.20)))
        + (0.257 * std::exp(-.803 * std::log(einc+1.20)))*theRandGauss->fire();
  febss = std::min(1.0,febss);
  febss = std::max(0.0,febss);

  G4double cogbss = 0.1;
  beta [3] = 1.0;
  alpha[3] = cogbss*beta[3];

  //energy to deposit
  energyToDeposit = std::max(0.05,fracEnergyDeposit)*einc;

  //fraction of energy to deposit by each component

  frac[0] = 1.0 - fracEnergyByFirstPi0;
  frac[1] = fracEnergyByFirstPi0 * (1.0 - fracEnergyByLatePi0);
  frac[2] = fracEnergyByFirstPi0 *        fracEnergyByLatePi0;
  frac[3] = febss;

  for (G4int i = 1; i < 4 ; i++) {
    frac[i] = std::max(0.0,frac[i]);
    frac[i] = std::min(1.0,frac[i]);
  }

  G4double Xc;
  for (G4int i = 0 ; i < showerType ; i++) {
    if ( alpha[i]/beta[i] < .01 ) alpha[i] = .01 * beta[i];
    if ( beta[i] > 50. ) {
      Xc = 50./beta[i];
      alpha[i] *= Xc;
      beta[i] *= Xc;
    }
    if ( alpha[i] > 20. ){
      Xc = 20./alpha[i];
      alpha[i] *= Xc;
      beta[i] *= Xc;
    }
  }

  // parameters for the lateral profile

  lateralPar[0] = 0.174;
  lateralPar[1] = std::max(0.0,0.407-0.061*logEinc);
  lateralPar[2] = 1.027 * (0.675 - .048*std::max(0.,logEinc));
  lateralPar[3] = 0.184 * lateralPar[2];

}

void GflashHadronShowerProfile::fillFluctuationVector(){

  const int mDim[Gflash::NMX]   = {3,3,6,6,9,9,9,9};

  for(G4int k = 0 ; k < Gflash::NMX ; k++) {

     const G4int dim = mDim[k];
     G4double **xr   = new G4double *[dim];
     G4double **xrho = new G4double *[dim];
     for(G4int j=0;j<dim;j++) {
  	xr[j]   = new G4double [dim];
  	xrho[j] = new G4double [dim];
     }

     for(G4int i = 0; i < dim; i++) {
  	for(G4int j = 0; j < i+1 ; j++) {
  	  xrho[i][j] = Gflash::rho[i+Gflash::ISTCOL[k]-1][j];
  	  xrho[j][i] = xrho[i][j];
  	}
     }
     
     doCholeskyReduction(xrho,xr,dim);

     for(G4int i = 0 ; i < dim ; i++) {
  	for (G4int j = 0 ; j < i+1 ; j++){
  	  RMX[Gflash::NSTRTR[k]+i*(i+1)/2 + j] = xr[i][j];
  	}
     }

     for(G4int j=0;j<dim;j++) delete [] xr[j];
     delete [] xr;
     for(G4int j=0;j<dim;j++) delete [] xrho[j];
     delete [] xrho;
  }
}

void GflashHadronShowerProfile::doCholeskyReduction(double **vv, double **cc, const int ndim) {

  G4double sumCjkSquare;
  G4double vjjLess;
  G4double sumCikjk;

  cc[0][0] = std::sqrt(vv[0][0]);

  for(G4int j=1 ; j < ndim ; j++) {
    cc[j][0] = vv[j][0]/cc[0][0];
  }

  for(G4int j=1 ; j < ndim ; j++) {

    sumCjkSquare = 0.0;
    for (G4int k=0 ; k < j ; k++) sumCjkSquare += cc[j][k]*cc[j][k];
    
    vjjLess =  vv[j][j] - sumCjkSquare;

    if ( vjjLess < 0. ) {
      std::cout << "GflashHadronShowerProfile::CholeskyReduction failed " << std::endl;
    }
    else {
      cc[j][j] = std::sqrt(std::fabs(vjjLess));
      
      for (G4int i=j+1 ; i < ndim ; i++) {
	sumCikjk = 0.;
	for(G4int k=0 ; k < j ; k++) sumCikjk += cc[i][k]*cc[j][k];
	cc[i][j] = (vv[i][j] - sumCikjk)/cc[j][j];
      }
    }
  }
}

void GflashHadronShowerProfile::samplingFluctuation(G4double &de, G4double einc){

  G4double spot[Gflash::NDET];

  for(G4int i = 0 ; i < Gflash::NDET ; i++) {
    spot[i] = std::pow(Gflash::SAMHAD[0][i],2) // resolution 
      + std::pow(Gflash::SAMHAD[1][i],2)/einc  // noisy
      + std::pow(Gflash::SAMHAD[2][i],2)*einc; // constant 
  }
  G4double ein = de * (energyToDeposit/einc);

  de = (ein > 0 ) ?  
    theRandGamma->fire(ein/spot[jCalorimeter],1.0)*spot[jCalorimeter] : ein;
}

G4bool GflashHadronShowerProfile::insideSampling(const G4ThreeVector pos) {
  G4bool issampling = false;

  if((jCalorimeter == Gflash::kHB) || (jCalorimeter == Gflash::kHE) ||            
     ((jCalorimeter == Gflash::kESPM) && ((pos.rho()/cm - 177.5) > 0)) ||     
     ((jCalorimeter == Gflash::kENCA) && ( fabs(pos.z()/cm - 391.95) > 0 ))) issampling = true;
  
  return issampling;
}

