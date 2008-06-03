#include "SimG4Core/GFlash/interface/GflashHadronShowerProfile.h"
#include "SimG4Core/GFlash/interface/GflashEnergySpot.h"
#include "SimG4Core/GFlash/interface/GflashHistogram.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/GenericFunctions/IncompleteGamma.hh"
#include "CLHEP/GenericFunctions/LogGamma.hh"
#include "Randomize.hh"

//#include "SimG4Core/GFlash/interface/GflashTrajectory.h"
#include "SimG4Core/GFlash/interface/GflashTrajectoryPoint.h"

#include <math.h>

GflashHadronShowerProfile::GflashHadronShowerProfile(G4Region* envelope, edm::ParameterSet parSet) : theParSet(parSet)
{
  showerType   = 0;
  jCalorimeter = Gflash::kNULL;
  theHelix = new GflashTrajectory;
  theHisto = GflashHistogram::instance();
  theBField = parSet.getParameter<double>("bField");

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

  //correllation and fluctuation matrix
  fillFluctuationVector();
}

GflashHadronShowerProfile::~GflashHadronShowerProfile()
{
  //  delete theGflashStep;
  delete theHelix;
  delete theRandGauss;
  delete theRandGamma;
}

Gflash::CalorimeterNumber GflashHadronShowerProfile::getCalorimeterNumber(const G4ThreeVector position)
{
  Gflash::CalorimeterNumber index = Gflash::kNULL;
  G4double eta = position.getEta();

  //central
  if (fabs(eta) < Gflash::EtaMax[Gflash::kESPM] || fabs(eta) < Gflash::EtaMax[Gflash::kHB]) {
    if(position.getRho() > Gflash::Rmin[Gflash::kESPM] && 
       position.getRho() < Gflash::Rmax[Gflash::kESPM] ) {
      index = Gflash::kESPM;
    }
    if(position.getRho() > Gflash::Rmin[Gflash::kHB] && 
       position.getRho() < Gflash::Rmax[Gflash::kHB]) {
      index = Gflash::kHB;
    }
  }
  //forward
  else if (fabs(eta) > Gflash::EtaMin[Gflash::kENCA] || fabs(eta) > Gflash::EtaMin[Gflash::kHE]) {
    if( fabs(position.getZ()) > Gflash::Zmin[Gflash::kENCA] &&  
	fabs(position.getZ()) < Gflash::Zmax[Gflash::kENCA] ) {
      index = Gflash::kENCA;
    }
    if( fabs(position.getZ()) > Gflash::Zmin[Gflash::kHE] &&  
	fabs(position.getZ()) < Gflash::Zmax[Gflash::kHE] ) {
      index = Gflash::kHE;
    }
  }
  return index;
}

void GflashHadronShowerProfile::hadronicParameterization(const G4FastTrack& fastTrack)
{
  // The skeleton of this method is based on the fortran code gfshow.F originally written  
  // by S. Peters and G. Grindhammer (also see NIM A290 (1990) 469-488), but longitudinal
  // parameterizations of hadron showers are significantly modified for the CMS calorimeter  

  // unit convention: energy in [GeV] and length in [cm]

  // maximum number of energy spots 
  const G4int    maxNumberOfSpots = 10000;  

  // low energy cutoff (unit in GeV)
  //  const G4double energyCutoff     = 0.01; 

  // intrinsic properties of hadronic showers (lateral shower profile)
  const G4double maxShowerDepthforR50 = 10.0;

  G4double rShower = 0.;
  G4double rGauss = theRandGauss->fire();

  // The shower starting point is the PostStepPoint of Hadronic Inelestic interaction;
  // see GflashHadronShowerModel::ModelTrigger
                                                                                                   
  G4double einc = fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV;
  G4ThreeVector positionShower = fastTrack.GetPrimaryTrack()->GetPosition()/cm;
  G4ThreeVector momentumShower = fastTrack.GetPrimaryTrack()->GetMomentum()/GeV;

  //find the calorimeter at the shower starting point
  jCalorimeter = getCalorimeterNumber(positionShower);

  //get all necessary parameters for hadronic shower profiles including energyToDeposit
  loadParameters(fastTrack);

  // The direction of shower is assumed to be along the showino trajectory 
  // inside the magnetic field;
  double charge = fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint()->GetCharge();
  theHelix->initializeTrajectory(momentumShower,positionShower,charge,theBField/tesla);

  //path Length from the origin to the shower starting point in cm

  G4double pathLength0 = 0;
  G4double transDepth = 0;

  // The step length left is the total path length from the shower starting point to
  // the maximum distance inside paramerized envelopes

  //distance to the end of HB/HB now 
  //@@@extend the trajectory outside bField and HO later
  G4double stepLengthLeft = 0.0;

  if(jCalorimeter == Gflash::kESPM || jCalorimeter == Gflash::kHB ) {
    pathLength0 = theHelix->getPathLengthAtRhoEquals(positionShower.getRho());
    stepLengthLeft = theHelix->getPathLengthAtRhoEquals(Gflash::Rmax[Gflash::kHB])
      - theHelix->getPathLengthAtRhoEquals(positionShower.getRho());
    if(showerType == 3 ) {
      transDepth = theHelix->getPathLengthAtRhoEquals(Gflash::Rmin[Gflash::kHB]) - pathLength0;
    }
  }
  else if (jCalorimeter == Gflash::kENCA || jCalorimeter == Gflash::kHE ) {
    pathLength0 = theHelix->getPathLengthAtZ(positionShower.getZ());
    stepLengthLeft = theHelix->getPathLengthAtRhoEquals(Gflash::Zmax[Gflash::kHE])
      - theHelix->getPathLengthAtRhoEquals(positionShower.getZ());
    if ( showerType ==7 ) {
      transDepth = theHelix->getPathLengthAtZ(Gflash::Zmin[Gflash::kHE]) - pathLength0;
    }
  }
  else { 
    //@@@extend for HF later
    stepLengthLeft = 200.0;
  }
  
  G4double pathLength  = pathLength0; // this will grow along the shower development

  // Limit number of spots to maxNumberOfSpots
  G4int numberOfSpots = std::max( 50, static_cast<int>(800.*std::log(einc)+50.));
  numberOfSpots = std::min(numberOfSpots,maxNumberOfSpots);

  // Spot energy to simulate sampling fluctuations (SampleEnergySpot) and
  // number of spots needed to fullfill geometry constraints(TotalNumberOfSpots):

  G4double  spotEnergy = energyToDeposit/numberOfSpots;

  // The step size of showino along the helix trajectory in cm unit
  const G4double divisionStep = 1.0; 
  G4double deltaStep = 0.0;
  G4double showerDepth = 0.0;


  G4int totalNumberOfSpots = 0;

  //empty energy spot vector for a new track
  aEnergySpotList.clear();

  double scaleLateral = 0.0;

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

    double heightProfile = longitudinalProfile(showerDepth,pathLength,transDepth);
    //    deltaEnergy =  longitudinalProfile(showerDepth)*divisionStep*energyToDeposit;    
    deltaEnergy =  heightProfile*divisionStep*energyToDeposit;    
    
    //@@@ When depthShower is inside Hcal, the sampling fluctuation for deposited
    //    energy will be treated in SD.  However we should put some scale factor 
    //    to relate the spot energy to the energy deposited in each geant4 step. 

    double hadronicFraction = 1.0;
    G4double fluctuatedEnergy = deltaEnergy;
    G4int nSpotsInStep = std::max(1,static_cast<int>(fluctuatedEnergy/spotEnergy));
    G4double sampleSpotEnergy = hadronicFraction*fluctuatedEnergy/nSpotsInStep;

    // Sampling fluctuations determine the number of spots:
    //    if (insideSampling(positionShower)) samplingFluctuation(fluctuatedEnergy,einc); 
    //    G4double hadSpotEnergy = std::max(0.,sampleSpotEnergy * hadronicFraction);
    //    G4double emSpotEnergy  = std::max(0.,sampleSpotEnergy - hadSpotEnergy);
    //    hadSpotEnergy *= Gflash::PBYMIP[jCalorimeter];

    
    //@@@this part of code may not be not need, but leave it for further consideration

    // Lateral shape and fluctuations

    double showerDepthR50 = std::min(showerDepth/20.7394, maxShowerDepthforR50);

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
    if(showerType == 4 || showerType == 8) {
      scaleLateral = (3.5+1.0*showerDepth)*Gflash::rMoliere[jCalorimeter];
    }
    else {
      //@@@need better division for showerDepth arosse the Hcal front face
      if(showerDepthR50 < 2.0 ) {
	scaleLateral = (5.5-0.4*std::log(einc))*Gflash::rMoliere[jCalorimeter];
      }
      else {
	scaleLateral = ( 14-1.5*std::log(einc))*Gflash::rMoliere[jCalorimeter];
      }
    }
    // region0 && inside Ecal: scaleLateral = (5.5-0.4*logEinc)*Gflash::rMoliere[jCalorimeter];
    // region0 && inside Hcal: scaleLateral = (14-1.5*logEinc)*Gflash::rMoliere[jCalorimeter];
    // region1                 

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
      theHelix->getGflashTrajectoryPoint(trajectoryPoint,pathLength+incrementPath);

      // actual spot position by adding a radial vector to a trajectoryPoint
      G4ThreeVector SpotPosition = trajectoryPoint.getPosition() +
        rShower*std::cos(azimuthalAngle)*trajectoryPoint.getOrthogonalUnitVector() +
        rShower*std::sin(azimuthalAngle)*trajectoryPoint.getCrossUnitVector();

      //convert unit of energy to geant4 default MeV
      //      eSpot.setEnergy((hadSpotEnergy+emSpotEnergy)*GeV);
      eSpot.setEnergy(sampleSpotEnergy*GeV);
      eSpot.setPosition(SpotPosition*cm);
      aEnergySpotList.push_back(eSpot);

      //@@@debugging histograms
      if(theHisto->getStoreFlag()) {
	theHisto->rshower->Fill(rShower);
	theHisto->lateralx->Fill(rShower*std::cos(azimuthalAngle));
	theHisto->lateraly->Fill(rShower*std::sin(azimuthalAngle));
	theHisto->gfhlongProfile->Fill(pathLength+incrementPath-pathLength0,positionShower.getRho(),eSpot.getEnergy()*GeV);
      }
    }
  }

}

void GflashHadronShowerProfile::loadParameters(const G4FastTrack& fastTrack)
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

  showerType = 0;

  G4double correctionAsDepth = 0.0;

  //central
  if (jCalorimeter == Gflash::kESPM || jCalorimeter == Gflash::kHB ) {

    G4double posRho = position.getRho();

    if(pos1 != std::string::npos) {
      showerType = 2;
    }
    else {
      if(jCalorimeter == Gflash::kESPM) {
	showerType = 3;
	if( posRho < 129.0 ) showerType = 1;
      }
      else showerType = 4;
    }

    if ( posRho < 150.0 ) {
      correctionAsDepth = 0.01-2.0/((posRho-150.)*(posRho-150.) +5.4*5.4);
    }
    else {
      correctionAsDepth = 0.03-2.0/((posRho-150.)*(posRho-150.) +4.7*4.7);
    }
  }
  //forward
  else if (jCalorimeter == Gflash::kENCA || jCalorimeter == Gflash::kHE) {
    if(pos2 != std::string::npos) {
      showerType = 6;
    }
    else {
      if(jCalorimeter == Gflash::kENCA) {
	showerType = 7;
	if(fabs(position.getZ()) < 330.0 ) showerType = 5;
      }
      else showerType = 8;
    }
    //@@@need z-dependent correction on the mean energy reponse
  }

  
  // total energy to deposite
  //@@@ need additional parameterization by the shower starting point
  G4double fractionEnergy  = 1.0;
  G4double sigmaEnergy = 0.0;

  if( showerType == 4 || showerType == 8) { 
    //Mip-like particle
    fractionEnergy = 0.7125 + 0.0812*std::tanh(0.9040*(std::log(einc) - 2.6307));
    sigmaEnergy = 0.0257/std::sqrt(einc) + 0.0734;
  }
  else {
    fractionEnergy = 0.7125 + 0.0812*std::tanh(0.9040*(std::log(einc) - 2.6307));
    sigmaEnergy = 0.0844/std::sqrt(einc) + 0.0592;
  }

  energyToDeposit = fractionEnergy*(1.0+correctionAsDepth)*einc*(1.0+sigmaEnergy*theRandGauss->fire());
  energyToDeposit = std::max(0.0,energyToDeposit);

  // parameters for the longitudinal profiles
  //@@@check longitudinal profiles of endcaps for possible varitations
  //@@@need to add fluctuation and correlation for individual shower

  longPar[0][0] = 1.41*std::max(0.0,-5.96481e-03 + 0.18231*std::tanh(0.55451*(std::log(einc)-0.458775))) ;
  longPar[0][1] = std::max(0.0,2.01611 + 1.77483 * std::tanh(0.75719*(std::log(einc) - 2.58172)));
  longPar[0][2] = std::max(0.0,0.21261 + 0.24168 * std::tanh(0.76962*(std::log(einc) - 2.11936)));
  longPar[0][3] = std::max(0.0,1.05577e-02 + 1.00807  * std::tanh(-6.31044e-04*(std::log(einc) - 4.60658)));
  longPar[0][4] = 0.87*std::max(0.0,1.19845e-01 + 6.87070e-02 * std::tanh(-8.23888e-01*(std::log(einc) - 2.90178)));
  longPar[0][5] = std::max(0.0,2.49694e+01 + 1.10258e+01 * std::tanh(6.16435e-01*(std::log(einc) - 3.56012)));

  longSigma[0][0] = 0.02;
  longSigma[0][1] = 0.16;
  longSigma[0][2] = 0.02;
  longSigma[0][3] = 0.01;
  longSigma[0][4] = 0.03;
  longSigma[0][5] = 2.50;
  
  longPar[1][0] = 0.1126;
  longPar[1][1] = 1.3857;
  longPar[1][2] = std::max(0.0,1.1353 + 0.4997*std::tanh(-0.6382*(std::log(einc) - 2.0035)));
  longPar[1][3] = 0.2300;
  longPar[1][4] = 3.5018;
  longPar[1][5] = std::max(0.0,0.6151 - 0.0561*std::log(einc));

  longSigma[1][0] = 0.01;
  longSigma[1][1] = 0.44;
  longSigma[1][2] = 0.01;
  longSigma[1][3] = 0.01;
  longSigma[1][4] = 0.20;
  longSigma[1][5] = 0.04;

  longPar[2][0] = std::max(0.0,-1.55624e+01+1.56831e+01*std::tanh(5.93651e-01*(std::log(einc) + 4.89902)));
  longPar[2][1] = std::max(0.0,7.28995e-01+ 7.71148e-01*std::tanh(4.77898e-01*(std::log(einc) - 1.69087)));
  longPar[2][2] = std::max(0.0,1.23387+ 7.34778e-01*std::tanh(-3.14958e-01*(std::log(einc) - 0.529206)));
  longPar[2][3] = std::max(0.0,1.02070e+02+1.01873e+02*std::tanh(-4.99805e-01*(std::log(einc) + 5.04012)));
  longPar[2][4] = std::max(0.0,3.59765+8.53358e-01*std::tanh( 8.47277e-01*(std::log(einc) - 3.36548)));
  longPar[2][5] = std::max(0.0,4.27294e-01+1.62535e-02*std::tanh(-2.26278*(std::log(einc) - 1.81308)));

  longSigma[2][0] = 0.01;
  longSigma[2][1] = 0.44;
  longSigma[2][2] = 0.01;
  longSigma[2][3] = 0.01;
  longSigma[2][4] = 0.20;
  longSigma[2][5] = 0.04;

  double normalZ[Gflash::NxN];
  for (int i = 0; i < Gflash::NxN ; i++) normalZ[i] = theRandGauss->fire();
  
  for(int k = 0 ; k < Gflash::NRegion ; k++) {
    for(int i = 0 ; i < Gflash::NxN ; i++) {
      double correlationSum = 0.0;
      for(int j = 0 ; j < Gflash::NxN ; j++) {
	correlationSum += correlationVector[Gflash::NStart[Gflash::NRegion]+(i+1)/2+j]*normalZ[i];
      }
      longPar[k][i] = std::max(0.0,longPar[k][i]+longSigma[k][i]*correlationSum);
    }
  }

  // parameters for the lateral profile

  lateralPar[0] = 0.20;
  lateralPar[1] = std::max(0.0,0.40 -0.06*std::log(einc));
  lateralPar[2] = 0.70 - 0.05*std::max(0.,std::log(einc));
  lateralPar[3] = 0.20 * lateralPar[2];
}

G4double GflashHadronShowerProfile::longitudinalProfile(G4double showerDepth, G4double pathLength, G4double transDepth){

  G4double heightProfile = 0;

  // Energy in a delta step (dz) = (energy to deposite)*[Gamma(z+dz)-Gamma(z)]*dz
  // where the incomplete Gamma function gives an intergrate probability of the longitudinal 
  // shower u[ to the shower depth (z).
  // Instead, we use approximated energy; energy in dz = (energy to deposite)*gamma(z)*dz
  // where gamma is the Gamma-distributed probability function

  Genfun::LogGamma lgam;
  GflashTrajectoryPoint tempPoint;
  theHelix->getGflashTrajectoryPoint(tempPoint,pathLength);

  double x = 0.0;
  //get parameters
  if(showerType == 1 || showerType == 2 ) {
    //    std::cout << " pathLength tempPoint.getPosition().getRho()=  "  << pathLength << " "  << tempPoint.getPosition().getRho() << std::endl;
    if(tempPoint.getPosition().getRho() < 150.0 ) { 
      x = showerDepth*longPar[0][2];
      heightProfile = longPar[0][0]*std::pow(x,longPar[0][1]-1.0)*std::exp(-x)/std::exp(lgam(longPar[0][1]))+longPar[0][3];
    }
    else if (tempPoint.getPosition().getRho() > Gflash::Rmin[Gflash::kHB] ){
      x = showerDepth;
      heightProfile = longPar[0][4]*std::exp(-x/longPar[0][5]);
      heightProfile *= Gflash::ScaleSensitive;
    }
    else heightProfile = 0.;
  }  
  else if(showerType == 5 || showerType == 6){
    //@@@use new parameterization for EE/HE
    if(std::abs(tempPoint.getPosition().getZ()) < Gflash::Zmin[Gflash::kENCA]+23.0 ) { 
      x = showerDepth*longPar[0][2];
      heightProfile = longPar[0][0]*std::pow(x,longPar[0][1]-1.0)*std::exp(-x)/std::exp(lgam(longPar[0][1]))+longPar[0][3];
    }
    else if (std::abs(tempPoint.getPosition().getZ()) > Gflash::Rmin[Gflash::kHE] ){
      x = showerDepth;
      heightProfile = longPar[0][4]*std::exp(-x/longPar[0][5]);
      heightProfile *= Gflash::ScaleSensitive;
    }
    else heightProfile = 0.;
  }  
  else if (showerType == 3 || showerType == 7 ) {
    //two gammas between crystal and Hcal
    if((showerDepth - transDepth) > 0.0) {
      double x1 = (showerDepth-transDepth)*longPar[1][2]/16.42;
      double x2 = (showerDepth-transDepth)*longPar[1][5]/1.49;

      heightProfile = longPar[1][3]*std::pow(x1,longPar[1][1]-1.0)*std::exp(-x1)/std::exp(lgam(longPar[1][1]))
	+ (1.0-longPar[1][3])*std::pow(x2,longPar[1][4]-1.0)*std::exp(-x2)/std::exp(lgam(longPar[1][4]));
      heightProfile = std::max(0.0,longPar[1][0]*heightProfile);
      heightProfile *= Gflash::ScaleSensitive;
    }
    else heightProfile = 0.;
  }
  else if (showerType == 4 || showerType == 8 ) {
    //two gammas inside Hcal
    double x1 = showerDepth*longPar[2][2]/16.42;
    double x2 = showerDepth*longPar[2][5]/1.49;
    heightProfile = longPar[2][3]*std::pow(x1,longPar[2][1]-1.0)*std::exp(-x1)/std::exp(lgam(longPar[2][1]))
                  + (1.0-longPar[2][3])*std::pow(x2,longPar[2][4]-1.0)*std::exp(-x2)/std::exp(lgam(longPar[2][4]));
    heightProfile = std::max(0.0,longPar[2][0]*heightProfile);
    heightProfile *= Gflash::ScaleSensitive;
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

void GflashHadronShowerProfile::fillFluctuationVector() {
  //  G4double RMX[186]; //21*6 = 186

  for(G4int k = 0 ; k < Gflash::NRegion ; k++) {
    const G4int dim = Gflash::NDim[k];
    G4double **xr   = new G4double *[dim];
    G4double **xrho = new G4double *[dim];
    
    for(G4int j=0;j<dim;j++) {
      xr[j]   = new G4double [dim];
      xrho[j] = new G4double [dim];
    }
    
    for(G4int i = 0; i < dim; i++) {
      for(G4int j = 0; j < i+1 ; j++) {
	xrho[i][j] = Gflash::rho[i+Gflash::NRegion*k][j];
	xrho[i][j] = Gflash::rho[i][j];
	xrho[j][i] = xrho[i][j];
      }
    }
    
    doCholeskyReduction(xrho,xr,dim);

    for(G4int i = 0 ; i < dim ; i++) {
      for (G4int j = 0 ; j < i+1 ; j++){
	correlationVector[Gflash::NStart[k]+i*(i+1)/2 + j] = xr[i][j];
      }
    }

    std::cout << "this should be calcuated at constructor" << std::endl;
    for(int i = 0; i < 21 ; i++) std::cout << correlationVector[i] << std::endl;
    
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
