#include "SimG4Core/GFlash/interface/GflashHadronShowerProfile.h"
#include "SimG4Core/GFlash/interface/GflashHistogram.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/GenericFunctions/IncompleteGamma.hh"
#include "CLHEP/GenericFunctions/LogGamma.hh"
#include "Randomize.hh"
#include "G4TransportationManager.hh"
#include "G4TouchableHandle.hh"
#include "G4VSensitiveDetector.hh"
#include "G4VPhysicalVolume.hh"

//#include "SimG4Core/GFlash/interface/GflashTrajectory.h"
#include "SimG4Core/GFlash/interface/GflashTrajectoryPoint.h"

#include <math.h>

GflashHadronShowerProfile::GflashHadronShowerProfile(G4Region* envelope, edm::ParameterSet parSet) : theParSet(parSet)
{
  showerType   = -1;
  jCalorimeter = Gflash::kNULL;
  theHelix = new GflashTrajectory;
  theGflashStep = new G4Step();
  theGflashTouchableHandle = new G4TouchableHistory();

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
}

GflashHadronShowerProfile::~GflashHadronShowerProfile()
{
  //  delete theGflashStep;
  delete theHelix;
  delete theRandGauss;
  delete theRandGamma;
  if(theGflashStep) delete theGflashStep;
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
  const G4int    maxNumberOfSpots = 100000;  

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

  // inside the magnetic field (tesla unit);
  double charge = fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint()->GetCharge();
  theHelix->initializeTrajectory(momentumShower,positionShower,charge,theBField);

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
    transDepth = theHelix->getPathLengthAtRhoEquals(Gflash::Rmin[Gflash::kHB]) - pathLength0;
  }
  else if (jCalorimeter == Gflash::kENCA || jCalorimeter == Gflash::kHE ) {
    pathLength0 = theHelix->getPathLengthAtZ(positionShower.getZ());
    stepLengthLeft = theHelix->getPathLengthAtRhoEquals(Gflash::Zmax[Gflash::kHE])
      - theHelix->getPathLengthAtRhoEquals(positionShower.getZ());
    transDepth = theHelix->getPathLengthAtZ(Gflash::Zmin[Gflash::kHE]) - pathLength0;
  }
  else { 
    //@@@extend for HF later
    stepLengthLeft = 200.0;
  }
  
  G4double pathLength  = pathLength0; // this will grow along the shower development

  // Limit number of spots to maxNumberOfSpots
  G4int numberOfSpots = getNumberOfSpots(einc);
  numberOfSpots = std::min(numberOfSpots,maxNumberOfSpots);

  // Spot energy to simulate sampling fluctuations (SampleEnergySpot) and
  // number of spots needed to fullfill geometry constraints(TotalNumberOfSpots):

  G4double  spotEnergy = energyToDeposit/numberOfSpots;

  // The step size of showino along the helix trajectory in cm unit
  const G4double divisionStep = 1.0; 
  G4double deltaStep = 0.0;
  G4double showerDepth = 0.0;

  G4int totalNumberOfSpots = 0;

  double scaleLateral = 0.0;

  Gflash::CalorimeterNumber whichCalor = jCalorimeter;

  // navigator and time information is needed for making a fake step
  theGflashNavigator = new G4Navigator();
  theGflashNavigator->SetWorldVolume(G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume());
  G4double timeGlobal = fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint()->GetGlobalTime();

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

    //trajectory point of showino along the shower depth (at the pathLength)
    GflashTrajectoryPoint trajectoryShowino;
    theHelix->getGflashTrajectoryPoint(trajectoryShowino,pathLength);

    // energy in this deltaStep along the longitudinal shower profile
    double deltaEnergy = 0.;

    //double heightProfile = longitudinalProfile(showerDepth,pathLength,transDepth);
    //@@@O.K, we need the better way of passing arguments here, use like this temporarily
    double heightProfile = longitudinalProfile(showerDepth,pathLength,transDepth,positionShower,einc);

    //get proper energy scale 
      
    whichCalor = getCalorimeterNumber(trajectoryShowino.getPosition());
    deltaEnergy =  heightProfile*divisionStep*energyScale[whichCalor];    

    //@@@ When depthShower is inside Hcal, the sampling fluctuation for deposited
    //    energy will be treated in SD.  However we should put some scale factor 
    //    to relate the spot energy to the energy deposited in each geant4 step. 

    double hadronicFraction = 1.0;
    G4double fluctuatedEnergy = deltaEnergy;
    G4int nSpotsInStep = std::max(1,static_cast<int>(fluctuatedEnergy/spotEnergy));
    G4double sampleSpotEnergy = hadronicFraction*fluctuatedEnergy/nSpotsInStep;

    //@@@@syjun trial consideration for Sampling Detector - only for the central detector
    //until we optimize the reduction scale in the number of Nspots
    // trajectoryPoint for along the longitudinal profile center

    G4double rhoShowino = (trajectoryShowino.getPosition()).getRho();

    if(rhoShowino < Gflash::Rmin[Gflash::kHB] ) {
      nSpotsInStep = static_cast<int>(nSpotsInStep/50);
      sampleSpotEnergy = sampleSpotEnergy*50.0;
    }
    else {
      nSpotsInStep = static_cast<int>(nSpotsInStep/5.0);
      sampleSpotEnergy = sampleSpotEnergy*5.0;
    }

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
    if(showerType == 3 || showerType == 7) {
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

    R50 *= scaleLateral;


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

      //@@@debugging histograms
      if(theHisto->getStoreFlag()) {
	theHisto->rshower->Fill(rShower);
	theHisto->lateralx->Fill(rShower*std::cos(azimuthalAngle));
	theHisto->lateraly->Fill(rShower*std::sin(azimuthalAngle));
      }

      sampleSpotEnergy *= GeV;
      SpotPosition *= cm;

      // to make a different time for each fake step. (+1.0 is arbitrary)
      timeGlobal += 0.0001*nanosecond;

      // fill equivalent changes to a (fake) step associated with a spot 

      theGflashStep->SetTrack(const_cast<G4Track*>(fastTrack.GetPrimaryTrack()));
      theGflashStep->GetPostStepPoint()->SetGlobalTime(timeGlobal);
      theGflashStep->GetPreStepPoint()->SetPosition(SpotPosition);
      theGflashStep->GetPostStepPoint()->SetPosition(SpotPosition);
      theGflashStep->GetPostStepPoint()->SetProcessDefinedStep(const_cast<G4VProcess*> (fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()));

      theGflashNavigator->LocateGlobalPointAndUpdateTouchableHandle(SpotPosition,G4ThreeVector(0,0,0),theGflashTouchableHandle, false);
      theGflashStep->GetPreStepPoint()->SetTouchableHandle(theGflashTouchableHandle);
      
      theGflashStep->SetTotalEnergyDeposit(sampleSpotEnergy);
    
      // Send G4Step information to Hit/Dig if the volume is sensitive

      G4VPhysicalVolume* aCurrentVolume = theGflashStep->GetPreStepPoint()->GetPhysicalVolume();
      if( aCurrentVolume == 0 ) continue;

      G4LogicalVolume* lv = aCurrentVolume->GetLogicalVolume();
      if(lv->GetRegion()->GetName() != "GflashRegion") continue;

      theGflashStep->GetPreStepPoint()->SetSensitiveDetector(aCurrentVolume->GetLogicalVolume()->GetSensitiveDetector());
      G4VSensitiveDetector* aSensitive = theGflashStep->GetPreStepPoint()->GetSensitiveDetector();

      if( aSensitive == 0 ) continue;
      aSensitive->Hit(theGflashStep);

    } // end of for spot iteration

  } // end of while for longitudinal integration

  delete theGflashNavigator;

}

void GflashHadronShowerProfile::loadParameters(const G4FastTrack& fastTrack)
{
  // Initialization of longitudinal and lateral parameters for 
  // hadronic showers. Simulation of the intrinsic fluctuations

  G4double einc = fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV;

  // type of hadron showers subject to the shower starting point (ssp)
  // showerType = -1 : default (invalid) 
  // showerType =  0 : ssp before EBRY (barrel crystal) 
  // showerType =  1 : ssp inside EBRY
  // showerType =  2 : ssp after  EBRY before HB
  // showerType =  3 : ssp inside HB
  // showerType =  4 : ssp before EFRY (endcap crystal) 
  // showerType =  5 : ssp inside EFRY 
  // showerType =  6 : ssp after  EFRY before HE
  // showerType =  7 : ssp inside HE
    
  G4TouchableHistory* touch = (G4TouchableHistory*)(fastTrack.GetPrimaryTrack()->GetTouchable());
  G4LogicalVolume* lv = touch->GetVolume()->GetLogicalVolume();
  std::size_t pos1 = lv->GetName().find("EBRY");
  std::size_t pos2 = lv->GetName().find("EFRY");

  G4ThreeVector position = fastTrack.GetPrimaryTrack()->GetPosition()/cm;

  showerType = -1;

  G4double correctionAsDepth = 0.0;

  //central
  if (jCalorimeter == Gflash::kESPM || jCalorimeter == Gflash::kHB ) {

    G4double posRho = position.getRho();

    if(pos1 != std::string::npos) {
      showerType = 1;
    }
    else {
      if(jCalorimeter == Gflash::kESPM) {
	showerType = 2;
	if( posRho < 129.0 ) showerType = 0;
      }
      else showerType = 3;
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
      showerType = 5;
    }
    else {
      if(jCalorimeter == Gflash::kENCA) {
	showerType = 6;
	if(fabs(position.getZ()) < 330.0 ) showerType = 4;
      }
      else showerType = 7;
    }
    //@@@need z-dependent correction on the mean energy reponse
  }

  // total energy to deposite
  //@@@ need additional parameterization by the shower starting point
  G4double fractionEnergy  = 1.0;
  G4double sigmaEnergy = 0.0;
  
  fractionEnergy = fTanh(einc,Gflash::fdep[0]);
  sigmaEnergy = sqrt(Gflash::fdep[1][0]*Gflash::fdep[1][0]/einc 
			+ Gflash::fdep[1][1]*Gflash::fdep[1][1]);

  energyToDeposit = fractionEnergy*(1.0+correctionAsDepth)*einc*(1.0+sigmaEnergy*theRandGauss->fire());
  energyToDeposit = std::max(0.0,energyToDeposit);

  // energy scale
  //@@@ need additional parameterization for forward detectors

  double energyMeanHcal = 0.0;
  double energySigmaHcal = 0.0;

  if(showerType == 0 || showerType == 1 || showerType == 4 || showerType == 5) {

    G4double r1 = 0.0;
    G4double r2 = 0.0;

    //@@@ need energy dependent parameterization and put relevant parameters into GflashNameSpace
    //@@@ put energy dependent energyRho based on tuning with testbeam data
    G4double energyRho =  -0.92; 

    do {
      r1 = theRandGauss->fire();
      energyScale[Gflash::kESPM] = einc*(fTanh(einc,Gflash::emscale[0]) + fTanh(einc,Gflash::emscale[2])*r1);
    }
    while (energyScale[Gflash::kESPM] > einc || energyScale[Gflash::kESPM] < 0.0);

    //@@@extend depthScale for HE 
    energyMeanHcal  = (fTanh(einc,Gflash::hadscale[0]) +
		       fTanh(einc,Gflash::hadscale[1])*depthScale(position.getRho(),129.,22.));
    energySigmaHcal = (fTanh(einc,Gflash::hadscale[2]) +
		       fTanh(einc,Gflash::hadscale[3])*depthScale(position.getRho(),129.,22.));

    do {
      r2 = theRandGauss->fire();
      energyScale[Gflash::kHB] = 
	exp(energyMeanHcal+energySigmaHcal*(energyRho*r1 + sqrt(1.0- energyRho*energyRho)*r2 ))-0.05*einc;
    }
    while (energyScale[Gflash::kHB] > einc);
  }
  else if(showerType == 2 || showerType == 6) { //between crystal and Hcal (gap)
    energyMeanHcal  = einc*(fTanh(einc,Gflash::gapscale[0]) +
                            fTanh(einc,Gflash::gapscale[1])*depthScale(position.getRho(),179.,28.));
    energySigmaHcal = einc*(fTanh(einc,Gflash::gapscale[2]) +
                            fTanh(einc,Gflash::gapscale[3])*depthScale(position.getRho(),179.,28.));

    do{
      energyScale[Gflash::kHB] = energyMeanHcal + energySigmaHcal*theRandGauss->fire();
    }
    while (energyScale[Gflash::kHB] < 0 || energyScale[Gflash::kHB] > einc);

  }
  else if(showerType == 3 || showerType == 7) { //Hcal response for mip-like pions (mip)
    //@@@ no depth dependency yet, but may introduce later if necessary
    energyMeanHcal  = einc*fTanh(einc,Gflash::mipscale[0]);
    energySigmaHcal = einc*fTanh(einc,Gflash::mipscale[2]);

    do{
      //@@@ need additional scale in sigma? -> tuning
      energyScale[Gflash::kHB] = energyMeanHcal + energySigmaHcal*theRandGauss->fire();
    }
    while (energyScale[Gflash::kHB] < 0 || energyScale[Gflash::kHB] > einc);
  }

  energyScale[Gflash::kENCA] = energyScale[Gflash::kESPM];
  energyScale[Gflash::kHE] = energyScale[Gflash::kHB];

  // parameters for the longitudinal profiles
  //@@@check longitudinal profiles of endcaps for possible varitations
  //correllation and fluctuation matrix of longitudinal parameters

  G4double *rhoHcal = new G4double [2*Gflash::NPar];
  G4double *correlationVectorHcal = new G4double [Gflash::NPar*(Gflash::NPar+1)/2];

  //for now, until we have a separate parameterization for Endcap 
  if(showerType>3) showerType -= 4;
  if(showerType==0) showerType = 1; //no separate parameterization before crystal

  for(int i = 0 ; i < 2*Gflash::NPar ; i++ ) {
    rhoHcal[i] = fTanh(einc,Gflash::rho[i + showerType*2*Gflash::NPar]);
  }

  correlationVectorHcal = getFluctuationVector(rhoHcal);

  G4double normalZ[Gflash::NPar];
  for (int i = 0; i < Gflash::NPar ; i++) normalZ[i] = theRandGauss->fire();
  
  for(int i = 0 ; i < Gflash::NPar ; i++) {
    double correlationSum = 0.0;
    for(int j = 0 ; j < i+1 ; j++) {
      correlationSum += correlationVectorHcal[i*(i+1)/2+j]*normalZ[j];
    }
    longHcal[i] = fTanh(einc,Gflash::par[i+showerType*Gflash::NPar]) +
                  fTanh(einc,Gflash::par[i+(4+showerType)*Gflash::NPar])*correlationSum;
  }

  delete [] rhoHcal;
  delete [] correlationVectorHcal;

  if(showerType == 0 || showerType == 1) {
    G4double *rhoEcal = new G4double [2*Gflash::NPar];
    G4double *correlationVectorEcal = new G4double [2*Gflash::NPar];
    for(int i = 0 ; i < 2*Gflash::NPar ; i++ ) rhoEcal[i] = fTanh(einc,Gflash::rho[i]);

    correlationVectorEcal = getFluctuationVector(rhoEcal);

    for (int i = 0; i < Gflash::NPar ; i++) normalZ[i] = theRandGauss->fire();
    for(int i = 0 ; i < Gflash::NPar ; i++) {
      double correlationSum = 0.0;
      for(int j = 0 ; j < i+1 ; j++) {
	correlationSum += correlationVectorEcal[i*(i+1)/2+j]*normalZ[j];
      }
      longEcal[i] = fTanh(einc,Gflash::par[i]) +
        fTanh(einc,Gflash::par[i+4*Gflash::NPar])*correlationSum;
    }
    delete [] rhoEcal;
    delete [] correlationVectorEcal;
  }

  // parameters for the lateral profile

  lateralPar[0] = 0.20;
  lateralPar[1] = std::max(0.0,0.40 -0.06*std::log(einc));
  lateralPar[2] = 0.70 - 0.05*std::max(0.,std::log(einc));
  lateralPar[3] = 0.20 * lateralPar[2];

}

G4double GflashHadronShowerProfile::longitudinalProfile(G4double showerDepth, G4double pathLength, G4double transDepth,
							const G4ThreeVector pos,G4double einc) {
  G4double heightProfile = 0;

  // Energy in a delta step (dz) = (energy to deposite)*[Gamma(z+dz)-Gamma(z)]*dz
  // where the incomplete Gamma function gives an intergrate probability of the longitudinal 
  // shower up to the shower depth (z).
  // Instead, we use approximated energy; energy in dz = (energy to deposite)*gamma(z)*dz
  // where gamma is the Gamma-distributed probability function

  GflashTrajectoryPoint tempPoint;
  theHelix->getGflashTrajectoryPoint(tempPoint,pathLength);

  if(showerType == 0 || showerType == 1 ) {
    if(tempPoint.getPosition().getRho() < 152.0 ) { 
      heightProfile = twoGammaProfile(longEcal,showerDepth,Gflash::kESPM);
      
    }
    else if (tempPoint.getPosition().getRho() > Gflash::Rmin[Gflash::kHB] ){
      heightProfile = twoGammaProfile(longHcal,showerDepth-transDepth,Gflash::kHB);
	//@@@put this in the scale, not in the profile
	//      heightProfile *= Gflash::ScaleSensitive;
    }
    else heightProfile = 0.;
  }  
  else if(showerType == 4 || showerType == 5){
    //@@@use new parameterization for EE/HE
    if(std::abs(tempPoint.getPosition().getZ()) < Gflash::Zmin[Gflash::kENCA]+23.0 ) { 
      heightProfile = twoGammaProfile(longEcal,showerDepth,Gflash::kENCA);
    }
    else if (std::abs(tempPoint.getPosition().getZ()) > Gflash::Rmin[Gflash::kHE] ){
      heightProfile = twoGammaProfile(longHcal,showerDepth-transDepth,Gflash::kHE);
    }
    else heightProfile = 0.;
  }  
  else if (showerType == 2 || showerType == 6 ) {
    //two gammas between crystal and Hcal
    if((showerDepth - transDepth) > 0.0) {
      heightProfile = twoGammaProfile(longHcal,showerDepth-transDepth,Gflash::kHB);
    }
    else heightProfile = 0.;
  }
  else if (showerType == 3 || showerType == 7 ) {
    //two gammas inside Hcal
      heightProfile = twoGammaProfile(longHcal,showerDepth,Gflash::kHB);
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

G4double* GflashHadronShowerProfile::getFluctuationVector(G4double *lowTriangle) {

    const G4int dim = Gflash::NPar;

    G4double *correlationVector = new G4double [dim*(dim+1)/2];

    G4double **xr   = new G4double *[dim];
    G4double **xrho = new G4double *[dim];
    
    for(G4int j=0;j<dim;j++) {
      xr[j]   = new G4double [dim];
      xrho[j] = new G4double [dim];
    }
    
    for(G4int i = 0; i < dim; i++) {
      for(G4int j = 0; j < i+1 ; j++) {
        if(j==i) xrho[i][j] = 1.0;
	else {
	  xrho[i][j] = lowTriangle[i*(i-1)/2 + j];
	  xrho[j][i] = xrho[i][j];
	}
      }
    }

    doCholeskyReduction(xrho,xr,dim);

    for(G4int i = 0 ; i < dim ; i++) {
      for (G4int j = 0 ; j < i+1 ; j++){
	correlationVector[i*(i+1)/2 + j] = xr[i][j];
      }
    }

    return correlationVector;
    
    for(G4int j=0;j<dim;j++) delete [] xr[j];
    delete [] xr;
    for(G4int j=0;j<dim;j++) delete [] xrho[j];
    delete [] xrho;
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

G4int GflashHadronShowerProfile::getNumberOfSpots(G4double einc) {
  //generator number of spots: energy dependent Gamma distribution of Nspots based on Geant4
  //replacing old parameterization of H1, 
  //G4int numberOfSpots = std::max( 50, static_cast<int>(80.*std::log(einc)+50.));

  G4int numberOfSpots = 0;
  G4double alphaNspots = 0.0;
  G4double betaNspots  = 0.0;

  if(showerType == 0 || showerType == 1 || showerType == 4 || showerType == 5 ) {
    alphaNspots = 1.37719e+01+1.27901e+01*std::tanh(5.83278e-01*(std::log(einc)-2.71949e+00));
    betaNspots  = 1.43212e-01-1.43340e-01*std::tanh(2.62593e-01*(std::log(einc)+8.94369e+00));
  }
  else if (showerType == 2 || showerType == 3 || showerType == 6 || showerType == 7 ) {
    alphaNspots = 7.27190e+00+2.63063e+00*std::tanh(2.44278e+00*(std::log(einc)-2.16815e+00));
    betaNspots  = 2.11525e-01-2.11634e-01*std::tanh(3.78206e-01*(std::log(einc)+6.06337e+00));
  }
  else {
    alphaNspots = 5;
    betaNspots  = 0.01;
  }

  //@@@need correlation and individual fluctuation on alphaNspots and betaNspots here:
  //evaluating covariance should be straight forward since the distribution is 'one' Gamma

  numberOfSpots = std::max(500,static_cast<int> (theRandGamma->fire(alphaNspots,betaNspots)));
  return numberOfSpots;
}

G4double GflashHadronShowerProfile::fTanh(G4double einc, const G4double *par) {
  double func = 0.0;
  if(einc>0.0) func = par[0]+par[1]*std::tanh(par[2]*(std::log(einc)-par[3]));
  return func;
}

G4double GflashHadronShowerProfile::depthScale(G4double ssp, G4double ssp0, G4double length) {
  double func = 0.0;
  if(length>0.0) func = std::pow((ssp-ssp0)/length,2.0);
  return func;
}

G4double GflashHadronShowerProfile::twoGammaProfile(G4double *longPar, G4double depth, Gflash::CalorimeterNumber kIndex) {
  G4double twoGamma = 0.0;
  twoGamma  = longPar[0]* gammaProfile(exp(longPar[1]),exp(longPar[2]),depth,Gflash::radLength[kIndex])
          +(1-longPar[0])*gammaProfile(exp(longPar[3]),exp(longPar[4]),depth,Gflash::intLength[kIndex]);
  return twoGamma;
}

G4double GflashHadronShowerProfile::gammaProfile(G4double alpha, G4double beta, G4double showerDepth, G4double lengthUnit) {
  double gamma = 0.0;
  if(alpha > 0 && beta > 0 && lengthUnit > 0) {
    Genfun::LogGamma lgam;
    double x = showerDepth*(beta/lengthUnit);
    gamma = (beta/lengthUnit)*std::pow(x,alpha-1.0)*std::exp(-x)/std::exp(lgam(alpha));
  }
  return gamma;
}

