#include "SimG4Core/GFlash/interface/GflashHadronShowerProfile.h"
#include "SimG4Core/GFlash/interface/GflashEnergySpot.h"
#include "SimG4Core/GFlash/interface/GflashHistogram.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/GenericFunctions/IncompleteGamma.hh"
#include "CLHEP/GenericFunctions/LogGamma.hh"
#include "Randomize.hh"

#include "SimG4Core/GFlash/interface/GflashTrajectory.h"
#include "SimG4Core/GFlash/interface/GflashTrajectoryPoint.h"

#include <math.h>

GflashHadronShowerProfile::GflashHadronShowerProfile(G4Region* envelope)
{
  jCalorimeter = Gflash::kNULL;
  showerType   = 0;

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
  // The skeleton of this method is based on the fortran code gfshow.F originally written  
  // by S. Peters and G. Grindhammer (also see NIM A290 (1990) 469-488), but longitudinal
  // parameterizations of hadron showers are significantly modified for the CMS calorimeter  

  // unit convention: energy in [GeV] and length in [cm]

  // maximum number of energy spots 
  const G4int    maxNumberOfSpots = 1500;  

  // low energy cutoff (unit in GeV)
  //  const G4double energyCutoff     = 0.01; 

  // intrinsic properties of hadronic showers (lateral shower profile)
  const G4double maxShowerDepthforR50 = 2.0;

  //@@@@@@@@@@@@@@@@@@@@@
  //  jCalorimeter = theMediaMap->getCalorimeterNumber(fastTrack);
  jCalorimeter = Gflash::kESPM;

  G4double rShower = 0.;
  G4double rGauss = theRandGauss->fire();

  // The shower starting point is the PostStepPoint of Hadronic Inelestic interaction;
  // see GflashHadronShowerModel::ModelTrigger
                                                                                                   
  G4double einc = fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV;
  G4ThreeVector positionShower = fastTrack.GetPrimaryTrack()->GetPosition()/cm;
  G4ThreeVector momentumShower = fastTrack.GetPrimaryTrack()->GetMomentum()/GeV;

  // The direction of shower is assumed to be along the showino trajectory 
  // inside the magnetic field;

  const G4double bField = 4.0*tesla; 
  double charge = fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint()->GetCharge();
  GflashTrajectory helix(momentumShower,positionShower,charge,bField/tesla);

  //path Length from the origin to the shower starting point in cm
  G4double pathLength0 = helix.getPathLengthAtRhoEquals(positionShower.getRho());
  G4double pathLength  = pathLength0; // this will grow along the shower development

  //get all necessary parameters for hadronic shower profiles
  loadLateralParameters(fastTrack);

  //@@@temporarily, may need an additional parameterization for the fraction of energy to deposit
  energyToDeposit = einc;

  // Limit number of spots to maxNumberOfSpots
  G4int numberOfSpots = std::max( 50, static_cast<int>(80.*std::log(einc)+50.));
  numberOfSpots = std::min(numberOfSpots,maxNumberOfSpots);

  // Spot energy to simulate sampling fluctuations (SampleEnergySpot) and
  // number of spots needed to fullfill geometry constraints(TotalNumberOfSpots):

  G4double  spotEnergy = energyToDeposit/numberOfSpots;

  // The step size of showino along the helix trajectory in cm unit
  const G4double divisionStep = 0.1; 
  G4double deltaStep = 0.0;
  G4double showerDepth = 0.0;

  // The step length left is the total path length from the shower starting point to
  // the maximum distance inside paramerized envelopes

  //distance to the end of HB/HB now 
  //@@@extend the trajectory outside bField and HO later
  G4double stepLengthLeft = 0.0;

  if(jCalorimeter == Gflash::kESPM || jCalorimeter == Gflash::kHB ) {
    stepLengthLeft = helix.getPathLengthAtRhoEquals(Gflash::Rmax[Gflash::kHB])
      - helix.getPathLengthAtRhoEquals(positionShower.getRho());
  }
  else if (jCalorimeter == Gflash::kENCA || jCalorimeter == Gflash::kHE ) {
    stepLengthLeft = helix.getPathLengthAtRhoEquals(Gflash::Zmax[Gflash::kHE])
      - helix.getPathLengthAtRhoEquals(positionShower.getZ());
  }
  else { 
    //@@@extend for HF later
    stepLengthLeft = 200.0;
  }

  G4int totalNumberOfSpots = 0;

  //empty energy spot vector for a new track
  aEnergySpotList.clear();

  //@@@debug histograms
  if(theHisto->getStoreFlag()) {
    theHisto->gfhssp->Fill(positionShower.getRho());
    theHisto->gfheinc->Fill(einc);
    theHisto->gfhsll->Fill(stepLengthLeft);
  }      

  double scaleLateral = 0.0;
  const double rMoliere = 2.19; //Moliere Radius in [cm]

  while(stepLengthLeft > 0.0) {

    // update shower depth and stepLengthLeft
    if ( stepLengthLeft < divisionStep ) {
      deltaStep = stepLengthLeft;
      stepLengthLeft  = 0.0;
    }
    else {
      deltaStep = divisionStep;
      stepLengthLeft -= deltaStep;
    }

    showerDepth += deltaStep;
    pathLength  += deltaStep;

    // energy in this deltaStep along the longitudinal shower profile
    double deltaEnergy = 0.;

    deltaEnergy =  longitudinalProfile(showerDepth)*divisionStep*energyToDeposit;    
    
    if(theHisto->getStoreFlag()) {
      theHisto->gfhlong->Fill(showerDepth,deltaEnergy);
    }      
    
    // Sampling fluctuations determine the number of spots:
    G4double fluctuatedEnergy = deltaEnergy;
    
    //@@@ sampling fluctuation when depthShower is inside Hcal
    //    if (insideSampling(positionShower)) samplingFluctuation(fluctuatedEnergy,einc); 

    G4int nSpotsInStep = std::max(1,static_cast<int>(deltaEnergy/spotEnergy));
    
    //@@@this part of code may not be not need, but leave it for further consideration
    double hadronicFraction = 1.0;
    G4double sampleSpotEnergy = fluctuatedEnergy/nSpotsInStep;
    G4double hadSpotEnergy = std::max(0.,sampleSpotEnergy * hadronicFraction);
    G4double emSpotEnergy  = std::max(0.,sampleSpotEnergy - hadSpotEnergy);

    hadSpotEnergy *= Gflash::PBYMIP[jCalorimeter];

    // Lateral shape and fluctuations

    double showerDepthR50 = std::min(showerDepth, maxShowerDepthforR50);
    double R50          = lateralPar[0] + lateralPar[1] * showerDepthR50;
    double varinanceR50 = std::pow((lateralPar[2] + lateralPar[3] * showerDepthR50) * R50, 2);

    // Simulation of lognormal distribution

    double sigmaSq  = std::log(varinanceR50/(R50*R50)+1.0);
    double sigmaR50 = std::sqrt(sigmaSq);
    double meanR50  = std::log(R50) - (sigmaSq/2.);

    R50    = std::exp(rGauss*sigmaR50 + meanR50);

    // Averaging lateral scale in terms of Moliere radius
    //    const G4double  rMoliere = Gflash::RLTHAD[jCalorimeter];

    //@@@this should be each spot basis
    scaleLateral = (5.5-0.4*std::log(einc))*rMoliere;
    // region0 && inside Ecal: scaleLateral = (5.5-0.4*logEinc)*rMoliere;
    // region0 && inside Hcal: scaleLateral = (14-1.5*logEinc)*rMoliere;
    // region1                 scaleLateral = (3.5+1.0*showerDepth)*rMoliere;

    R50 *= scaleLateral;

    GflashEnergySpot eSpot;

    for (G4int ispot = 0 ;  ispot < nSpotsInStep ; ispot++) {

      totalNumberOfSpots++;

      // Smearing in r according to f(r)= 2.*r*R50**2/(r**2+R50**2)**2
      G4double rnunif = G4UniformRand();
      G4double rxPDF  = std::sqrt(rnunif/(1.-rnunif));
      rShower  = R50 * rxPDF;

      // Uniform smearing in phi, for 66% of lateral containm.
      G4double azimuthalAngle = 0.0; 

      azimuthalAngle = twopi*G4UniformRand(); 

      // Compute global position of generated spots with taking into account magnetic field
      // Divide deltaStep into nSpotsInStep and give a spot a global position
      G4double incrementPath = (deltaStep/nSpotsInStep)*(ispot+0.5 - 0.5*nSpotsInStep);

      // trajectoryPoint give a spot an imaginary point along the shower development
      GflashTrajectoryPoint trajectoryPoint;
      helix.getGflashTrajectoryPoint(trajectoryPoint,pathLength+incrementPath);

      // actual spot position by adding a radial vector to a trajectoryPoint
      G4ThreeVector SpotPosition = trajectoryPoint.getPosition() +
        rShower*std::cos(azimuthalAngle)*trajectoryPoint.getOrthogonalUnitVector() +
        rShower*std::sin(azimuthalAngle)*trajectoryPoint.getCrossUnitVector();

      //convert unit of energy to geant4 default MeV
      eSpot.setEnergy((hadSpotEnergy+emSpotEnergy)*GeV);
      eSpot.setPosition(SpotPosition);
      aEnergySpotList.push_back(eSpot);

      //@@@debugging histograms
      if(theHisto->getStoreFlag()) {
	theHisto->rshower->Fill(rShower);
	theHisto->lateralx->Fill(rShower*std::cos(azimuthalAngle));
	theHisto->lateraly->Fill(rShower*std::sin(azimuthalAngle));
	theHisto->gfhlongProfile->Fill(pathLength+incrementPath-pathLength0,positionShower.getRho(),eSpot.getEnergy());
      }
    }
  }

}

void GflashHadronShowerProfile::loadLateralParameters(const G4FastTrack& fastTrack)
{
  // Initialization of longitudinal and lateral parameters for 
  // hadronic showers. Simulation of the intrinsic fluctuations

  G4double einc = fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV;

  // type of hadron showers subject to the shower starting point (ssp)
  // showerType =  0 : default (invalid) 
  // showerType =  1 : ssp before EB
  // showerType =  2 : ssp inside EB
  // showerType =  3 : ssp after  EB before HB
  // showerType =  4 : ssp inside HB
  // showerType =  5 : ssp before EE 
  // showerType =  6 : ssp inside EE 
  // showerType =  7 : ssp after  EE before HE
  // showerType =  8 : ssp inside HE
    
  G4TouchableHistory* touch = (G4TouchableHistory*)(fastTrack.GetPrimaryTrack()->GetTouchable());
  G4LogicalVolume* lv = touch->GetVolume()->GetLogicalVolume();
  std::size_t pos1 = lv->GetName().find("EBRY");
  std::size_t pos2 = lv->GetName().find("EFRY");

  G4ThreeVector position = fastTrack.GetPrimaryTrack()->GetPosition()/cm;
  G4double eta = fastTrack.GetPrimaryTrack()->GetMomentum().pseudoRapidity() ;

  //central
  if (fabs(eta) < Gflash::EtaMax[Gflash::kESPM] ) {
    if(pos1 != std::string::npos) {
      showerType = 2;
    }
    else {
      if(position.getRho() < Gflash::Rmax[Gflash::kESPM] ) showerType = 3;
      if(position.getRho() < 129.0 ) showerType = 1;
      else showerType = 4;
    }
  }
  //forward
  else if (fabs(eta) > Gflash::EtaMin[Gflash::kENCA] ) {
    if(pos2 != std::string::npos) {
      showerType = 6;
    }
    else {
      if(fabs(position.getZ()) < Gflash::Zmax[Gflash::kENCA] ) showerType = 7;
      if(fabs(position.getZ()) < 330.0 ) showerType = 5;
      else showerType = 8;
    }
  }
  else {
    showerType = 0;
  }

  // parameters for the longitudinal profiles

  longPar1[0] = std::max(0.05,-5.96481e-03 + 0.18231* std::tanh(0.55451*std::log(einc)-0.458775)) ;
  longPar1[1] = std::max(0.01,2.01611 + 1.77483 * std::tanh(0.75719*std::log(einc) - 2.58172));
  longPar1[2] = std::max(0.01,0.21261 + 0.24168 * std::tanh(0.76962*std::log(einc) - 2.11936));
  longPar1[3] = std::max(0.00,1.05577e-02 + 1.00807  * std::tanh(-6.31044e-04*std::log(einc) - 4.60658));
  longPar1[4] = std::max(0.01,1.19845e-01 + 6.87070e-02 * std::tanh(-8.23888e-01*std::log(einc) - 2.90178));
  longPar1[5] = std::max(0.00,2.49694e+01 + 1.10258e+01 * std::tanh(6.16435e-01*std::log(einc) - 3.56012));
    
  longPar2[0] = std::max(0.01,-1.55624e+01+1.56831e+01*std::tanh(5.93651e-01*std::log(einc) + 4.89902e+00));
  longPar2[1] = std::max(0.01,7.28995e-01+ 7.71148e-01*std::tanh(4.77898e-01*std::log(einc) - 1.69087e+00));
  longPar2[2] = std::max(0.01,1.23387e+00+ 7.34778e-01*std::tanh(-3.14958e-01*std::log(einc) - 5.29206e-01));
  longPar2[3] = std::max(0.01,1.02070e+02+1.01873e+02*std::tanh(-4.99805e-01*std::log(einc) + 5.04012e+00));
  longPar2[4] = std::max(0.01,3.59765e+00+8.53358e-01*std::tanh( 8.47277e-01*std::log(einc) - 3.36548e+00));
  longPar2[5] = std::max(0.01,4.27294e-01+1.62535e-02*std::tanh(-2.26278e+00*std::log(einc) - 1.81308e+00));

  // parameters for the lateral profile

  lateralPar[0] = 0.20;
  lateralPar[1] = std::max(0.0,0.40 -0.06*std::log(einc));
  lateralPar[2] = 0.70 - 0.05*std::max(0.,std::log(einc));
  lateralPar[3] = 0.20 * lateralPar[2];
}

G4double GflashHadronShowerProfile::longitudinalProfile(G4double showerDepth){

  G4double heightProfile = 0;

  // Energy in a delta step (dz) = (energy to deposite)*[Gamma(z+dz)-Gamma(z)]*dz
  // where the incomplete Gamma function gives an intergrate probability of the longitudinal 
  // shower u[ to the shower depth (z).
  // Instead, we use approximated energy; energy in dz = (energy to deposite)*gamma(z)*dz
  // where gamma is the Gamma-distributed probability function

  Genfun::LogGamma lgam;

  //get parameters
  if((showerType == 1 || showerType == 2) || (showerType == 5 || showerType == 6)) {
    double x = showerDepth*longPar1[2];
    if(showerDepth <23.0) { 
      heightProfile = longPar1[0]*std::pow(x,longPar1[1]-1.0)*std::exp(-x)/std::exp(lgam(longPar1[1])) + longPar1[3];
    }
    else {
      heightProfile = longPar1[4]*std::exp(-x/longPar1[5]);
    }
  }  
  else if ((showerType == 3 || showerType == 7 ) || (showerType == 4 || showerType == 8 )) {
    //two gammas
    double x1 = showerDepth*longPar2[2]/16.42;;
    double x2 = showerDepth*longPar2[5]/1.49;
    heightProfile = longPar2[0]*std::pow(x1,longPar2[1]-1.0)*std::exp(-x1)/std::exp(lgam(longPar2[1]));
                  + longPar2[3]*std::pow(x2,longPar2[4]-1.0)*std::exp(-x2)/std::exp(lgam(longPar2[4]));
  }
  else {
    heightProfile = 0;
  }

  return heightProfile;
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

