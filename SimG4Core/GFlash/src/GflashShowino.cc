#include "SimG4Core/GFlash/interface/GflashShowino.h"
#include "SimG4Core/GFlash/interface/GflashTrajectoryPoint.h"
#include "G4TouchableHandle.hh"

#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

#include "DataFormats/Math/interface/LorentzVector.h"

GflashShowino::GflashShowino() 
  : theEnergy(0), thePathLengthAtShower(0), thePositionAtShower(0,0,0), theStepLengthToOut(0), 
    theStepLengthToHcal(0), thePathLength(0), theGlobalTime(0), thePosition(0,0,0), 
    theEnergyDeposited(0), theDepthAtShower(0)
{
  theHelix =  new GflashTrajectory;;
}

GflashShowino::~GflashShowino() 
{
  delete theHelix;
}

void GflashShowino::initialize(const G4FastTrack& fastTrack, G4double magneticField)
{
  // initialization for the standard Gflash with G4FastTrack

  G4ThreeVector momentum = fastTrack.GetPrimaryTrack()->GetMomentum()/GeV;
  G4double charge = fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint()->GetCharge();

  theEnergy = fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV;
  theGlobalTime = fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint()->GetGlobalTime();
  thePositionAtShower = fastTrack.GetPrimaryTrack()->GetPosition()/cm;

  //simulate starting postion of the shower

  thePosition = thePositionAtShower;

  //set shower type depending on where is the shower starting point
  setShowerType(fastTrack);

  // inside the magnetic field (tesla unit);
  theHelix->initializeTrajectory(momentum,thePosition,charge,magneticField);

  evaluateLengths();

  theEnergyDeposited = 0.0;
}

void GflashShowino::initialize(const FSimTrack& fastTrack, G4double magneticField, const RandomEngine* randomEngine)
{
  //initialization for FastSimulation with FSimTrack

  XYZTLorentzVector fmom = fastTrack.momentum();
  G4ThreeVector momentum(fmom.X(),fmom.Y(),fmom.Z());

  float charge = fastTrack.charge();

  theEnergy = fastTrack.ecalEntrance().e();
  //hit time is not necessary for FastSim, but assign a dummy time to it 
  theGlobalTime = 150.0*nanosecond; 

  //the entrance vertex on ecal (FastSim): r=129cm for barrel and z=
  XYZTLorentzVector fpos = fastTrack.ecalEntrance().vertex();
  thePositionAtShower.set(fpos.X(),fpos.Y(),fpos.Z());

  // inside the magnetic field (tesla unit);
  theHelix->initializeTrajectory(momentum,thePositionAtShower,charge,magneticField);

  //determine the shower starting point (ssp):
  //the position at the entrance + the mean free path till the inelastic interaction inside calo

  //the pathLength at the reference (r=123.8 for barrel and z=304.5 for endcap)
  G4double pathLengthOnEcal = 0;
  theDepthAtShower = 0.0;

  //effective interaction length fitter to ssp from Geant4
  G4double effectiveLambda = 0.0;
  if(theEnergy > 0.0 &&  theEnergy < 15) {
    effectiveLambda = 24.6+2.6*std::tanh(3.0*(std::log(theEnergy)-1.43));
  }
  else {
    effectiveLambda = 28.4+1.20*std::tanh(1.5*(std::log(theEnergy)-4.3));
  }

  //fraction before the crystal, but inside Ecal
  G4double frac_ssp1 = 1.5196e-01+1.3300e-01*tanh(-4.6971e-01*(std::log(theEnergy)+2.4162e+00));
  //fraction after the crystal, but before Hcal
  G4double frac_ssp2 = 2.8310e+00+2.6766e+00*tanh(-4.8068e-01*(std::log(theEnergy)+3.4857e+00));

  if(fastTrack.onEcal() == 1 ) {
    G4double rhoTemp = Gflash::ROffCrystalEB + Gflash::LengthCrystalEB*std::sin(thePositionAtShower.getTheta());
    pathLengthOnEcal  = theHelix->getPathLengthAtRhoEquals(Gflash::Rmin[Gflash::kESPM]);
    G4double pathLengthAt1 = theHelix->getPathLengthAtRhoEquals(Gflash::Rmin[Gflash::kESPM]+ Gflash::ROffCrystalEB );
    G4double pathLengthAt2 = theHelix->getPathLengthAtRhoEquals(Gflash::Rmin[Gflash::kESPM]+ rhoTemp );
    G4double pathLengthAt3 = theHelix->getPathLengthAtRhoEquals(Gflash::Rmin[Gflash::kHB]);
    
    ///fraction before the crystal, but inside Ecal
    if(randomEngine->flatShoot() < frac_ssp1 ) {
      theDepthAtShower = (pathLengthAt1-pathLengthOnEcal)*randomEngine->flatShoot();
    }
    else { 
      //inside the crystal
      theDepthAtShower = (pathLengthAt1-pathLengthOnEcal) - effectiveLambda*log(randomEngine->flatShoot());
      //after the crystal
      if(theDepthAtShower > (pathLengthAt2 - pathLengthOnEcal) ) { 
	//before Hcal
	if(randomEngine->flatShoot() < frac_ssp2 ) {
	  theDepthAtShower = (pathLengthAt2 - pathLengthOnEcal) + (pathLengthAt3-pathLengthAt2)*randomEngine->flatShoot();
	}
	//inside Hcal
	else {
	  theDepthAtShower = (pathLengthAt3 - pathLengthOnEcal) - effectiveLambda*log(randomEngine->flatShoot());
	}
      }
    }
  }
  else if(fastTrack.onEcal() == 2 ) {
    G4double zTemp = Gflash::ZOffCrystalEE + Gflash::LengthCrystalEE;
    pathLengthOnEcal   = theHelix->getPathLengthAtZ(Gflash::Zmin[Gflash::kENCA]);
    G4double pathLengthAt1 = theHelix->getPathLengthAtZ(Gflash::Zmin[Gflash::kENCA] + Gflash::ZOffCrystalEE);
    G4double pathLengthAt2 = theHelix->getPathLengthAtZ(Gflash::Zmin[Gflash::kENCA] + zTemp);
    G4double pathLengthAt3 = theHelix->getPathLengthAtZ(Gflash::Zmin[Gflash::kHE]);
    
    if(randomEngine->flatShoot() <  frac_ssp1 ) {
      theDepthAtShower = (pathLengthAt1-pathLengthOnEcal)*randomEngine->flatShoot();
    }
    else {
      theDepthAtShower = (pathLengthAt1-pathLengthOnEcal)-effectiveLambda*std::log(randomEngine->flatShoot());
      
      if(theDepthAtShower > (pathLengthAt2 - pathLengthOnEcal) ) {
	if(randomEngine->flatShoot()<  frac_ssp2 ) {
	  theDepthAtShower = (pathLengthAt2 - pathLengthOnEcal) +(pathLengthAt3 - pathLengthAt2)*randomEngine->flatShoot();
	}
	else {
	  theDepthAtShower = (pathLengthAt3 - pathLengthOnEcal)-effectiveLambda*std::log(randomEngine->flatShoot());
	}
      }
    }
  }
  else if(fastTrack.onVFcal() == 2 ) { 
    //@@@extend for HF later
    pathLengthOnEcal = theHelix->getPathLengthAtZ(Gflash::Zmin[Gflash::kHF]);
    theDepthAtShower = 0.0;
  }
  
  G4double pathLength = pathLengthOnEcal + theDepthAtShower;

  GflashTrajectoryPoint trajectoryPoint;
  theHelix->getGflashTrajectoryPoint(trajectoryPoint,pathLength);  
  thePositionAtShower = trajectoryPoint.getPosition();

  //set the initial showino position at the shower starting position  
  thePosition = thePositionAtShower;

  //set shower type depending on where is the shower starting point
  setShowerType(fastTrack);

  std::cout << "r z pathLengthOnEcal theDepthAtShower pathLength " 
	    << thePosition.getRho() << " " << thePosition.getZ() << " " 
	    << pathLengthOnEcal << " " << theDepthAtShower << " " << pathLength << std::endl;

  std::cout << "p (r,z) q e t B showerType " << momentum 
            << " (" << thePositionAtShower.getRho() << "," << thePositionAtShower.getZ() << ") " 
	    << charge << " " << theEnergy << " " << theGlobalTime << " "
	    << magneticField << " " << getShowerType() << std::endl;

  evaluateLengths();

  theEnergyDeposited = 0.0;

}

void GflashShowino::setShowerType(const FSimTrack& fastTrack)
{
  G4int showerType = -1;

  G4double rhoBackEB = Gflash::Rmin[Gflash::kESPM] + Gflash::ROffCrystalEB 
                     + Gflash::LengthCrystalEB*std::sin(thePositionAtShower.getTheta());

  if(fastTrack.onEcal() == 1 || fastTrack.onHcal() == 1) { 
    //central
    G4double posRho = thePositionAtShower.getRho();

    if(posRho < Gflash::Rmin[Gflash::kESPM]+ Gflash::ROffCrystalEB ) {
      showerType = 0;
    }
    else if(posRho < rhoBackEB ) {
      showerType = 1;
    }
    else if (posRho < Gflash::Rmin[Gflash::kHB]) {
      showerType = 2;
    }
    else {
      showerType = 3;
    }
  }
  else if(fastTrack.onEcal() == 2 || fastTrack.onHcal() == 2) {
    //endcap
    G4double posZ = std::fabs(thePositionAtShower.getZ());

    if(posZ < Gflash::Zmin[Gflash::kENCA] + Gflash::ZOffCrystalEE) {
      showerType = 4;
    }
    else if(posZ < Gflash::Zmin[Gflash::kENCA] + Gflash::ZOffCrystalEE + Gflash::LengthCrystalEE) {
      showerType = 5;
    }
    else if (posZ < Gflash::Zmin[Gflash::kHE]) {
      showerType = 6;
    }
    else {
      showerType = 7;
    }
  }

  theShowerType = showerType;
}

void GflashShowino::setShowerType(const G4FastTrack& fastTrack)
{
  // Initialization of longitudinal and lateral parameters for 
  // hadronic showers. Simulation of the intrinsic fluctuations

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

  std::size_t pos1  = lv->GetName().find("EBRY");
  std::size_t pos11 = lv->GetName().find("EWAL");
  std::size_t pos12 = lv->GetName().find("EWRA");
  std::size_t pos2  = lv->GetName().find("EFRY");

  G4ThreeVector position = fastTrack.GetPrimaryTrack()->GetPosition()/cm;
  Gflash::CalorimeterNumber kCalor = Gflash::getCalorimeterNumber(position);

  G4int showerType = -1;

  //central
  if (kCalor == Gflash::kESPM || kCalor == Gflash::kHB ) {

    G4double posRho = position.getRho();

    if(pos1 != std::string::npos || pos11 != std::string::npos || pos12 != std::string::npos ) {
      showerType = 1;
    }
    else {
      if(kCalor == Gflash::kESPM) {
	showerType = 2;
	if( posRho < Gflash::Rmin[Gflash::kESPM]+ Gflash::ROffCrystalEB ) showerType = 0;
      }
      else showerType = 3;
    }

  }
  //forward
  else if (kCalor == Gflash::kENCA || kCalor == Gflash::kHE) {
    if(pos2 != std::string::npos) {
      showerType = 5;
    }
    else {
      if(kCalor == Gflash::kENCA) {
	showerType = 6;
	if(fabs(position.getZ()) < Gflash::Zmin[Gflash::kENCA] + Gflash::ZOffCrystalEE) showerType = 4;
      }
      else showerType = 7;
    }
    //@@@need z-dependent correction on the mean energy reponse
  }

  theShowerType = showerType;
}

void GflashShowino::updateShowino(G4double deltaStep)
{
  thePathLength += deltaStep;
  //trajectory point of showino along the shower depth at the pathLength
  GflashTrajectoryPoint trajectoryShowino;
  theHelix->getGflashTrajectoryPoint(trajectoryShowino,thePathLength);

  thePosition = trajectoryShowino.getPosition();
  theGlobalTime +=  (theEnergy/100.0)*deltaStep/3.0000e+10; //@@@calculate exact time change
}

void GflashShowino::evaluateLengths() {
  Gflash::CalorimeterNumber kCalor = Gflash::getCalorimeterNumber(thePosition);

  //thePathLengthAtShower: path Length from the origin to the shower starting point in cm
  //theStepLengthToOut: the total path length from the starting point of
  //                    shower to the maximum distance inside paramerized envelopes
  
  if(kCalor == Gflash::kESPM || kCalor == Gflash::kHB ) {
    thePathLengthAtShower = theHelix->getPathLengthAtRhoEquals(thePosition.getRho());
    theStepLengthToOut  = theHelix->getPathLengthAtRhoEquals(Gflash::Rmax[Gflash::kHB]) - thePathLengthAtShower;
    theStepLengthToHcal = theHelix->getPathLengthAtRhoEquals(Gflash::Rmin[Gflash::kHB]) - thePathLengthAtShower;
  }
  else if (kCalor == Gflash::kENCA || kCalor == Gflash::kHE ) {
    thePathLengthAtShower = theHelix->getPathLengthAtZ(thePosition.getZ());
    theStepLengthToOut  = theHelix->getPathLengthAtZ(Gflash::Zmax[Gflash::kHE]) - thePathLengthAtShower;
    theStepLengthToHcal = theHelix->getPathLengthAtZ(Gflash::Zmin[Gflash::kHE]) - thePathLengthAtShower;
  }
  else { 
    //@@@extend for HF later
    theStepLengthToOut = 200.0;
  }

  thePathLength = thePathLengthAtShower;

}

