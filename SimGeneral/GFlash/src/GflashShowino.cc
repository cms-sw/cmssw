#include "SimGeneral/GFlash/interface/GflashShowino.h"
#include <CLHEP/Random/Randomize.h>

GflashShowino::GflashShowino()
    : theShowerType(-1),
      theEnergy(0),
      thePositionAtShower(0, 0, 0),
      thePathLengthAtShower(0),
      thePathLengthOnEcal(0),
      theStepLengthToHcal(0),
      theStepLengthToOut(0),
      thePathLength(0),
      theGlobalTime(0),
      thePosition(0, 0, 0),
      theEnergyDeposited(0) {
  theHelix = new GflashTrajectory;
}

GflashShowino::~GflashShowino() { delete theHelix; }

void GflashShowino::initialize(int showerType,
                               double energy,
                               double globalTime,
                               double charge,
                               Gflash3Vector &position,
                               Gflash3Vector &momentum,
                               double magneticField) {
  theEnergy = energy;
  theGlobalTime = globalTime;
  theEnergyDeposited = 0.0;

  // inside the magnetic field (tesla unit);
  theHelix->initializeTrajectory(momentum, position, charge, magneticField);

  if (showerType < 100) {
    thePositionAtShower = position;
    thePosition = thePositionAtShower;
    theShowerType = showerType;

  } else {
    // this input is from FastSimulation
    // 1. simulate the shower starting position
    thePositionAtShower = simulateFirstInteractionPoint(showerType, position);
    thePosition = thePositionAtShower;

    // 2. find shower type depending on where is the shower starting point
    theShowerType = Gflash::findShowerType(thePositionAtShower);
  }

  evaluateLengths();
}

void GflashShowino::updateShowino(double deltaStep) {
  thePathLength += deltaStep;
  // trajectory point of showino along the shower depth at the pathLength
  GflashTrajectoryPoint trajectoryShowino;
  theHelix->getGflashTrajectoryPoint(trajectoryShowino, thePathLength);

  thePosition = trajectoryShowino.getPosition();

  theGlobalTime += (theEnergy / 100.0) * deltaStep / 30.0;  //@@@calculate exact time change in nsec
}

void GflashShowino::evaluateLengths() {
  // thePathLengthAtShower: path Length from the origin to the shower starting
  // point in cm theStepLengthToOut: the total path length from the starting
  // point of
  //                    shower to the maximum distance inside paramerized
  //                    envelopes
  double eta = thePosition.getEta();

  if (std::fabs(eta) < Gflash::EtaMax[Gflash::kESPM]) {
    thePathLengthOnEcal = theHelix->getPathLengthAtRhoEquals(Gflash::RFrontCrystalEB);
    thePathLengthAtShower = theHelix->getPathLengthAtRhoEquals(thePosition.getRho());
    double pathLengthAtHcalBack = theHelix->getPathLengthAtRhoEquals(Gflash::Rmax[Gflash::kHB]);
    if (pathLengthAtHcalBack > 0) {
      theStepLengthToOut = std::min(300., pathLengthAtHcalBack - thePathLengthAtShower);
    } else {
      theStepLengthToOut = 200.;
    }
    theStepLengthToHcal = theHelix->getPathLengthAtRhoEquals(Gflash::Rmin[Gflash::kHB]) - thePathLengthAtShower;

  } else if (std::fabs(eta) < Gflash::EtaMax[Gflash::kENCA]) {
    double zsign = (eta > 0) ? 1.0 : -1.0;
    thePathLengthOnEcal = theHelix->getPathLengthAtZ(zsign * Gflash::ZFrontCrystalEE);
    thePathLengthAtShower = theHelix->getPathLengthAtZ(thePosition.getZ());
    theStepLengthToOut =
        std::min(300., theHelix->getPathLengthAtZ(zsign * Gflash::Zmax[Gflash::kHE]) - thePathLengthAtShower);
    theStepLengthToHcal = theHelix->getPathLengthAtZ(zsign * Gflash::Zmin[Gflash::kHE]) - thePathLengthAtShower;
  } else {
    //@@@extend for HF later
    theStepLengthToOut = 200.0;
  }

  thePathLength = thePathLengthAtShower;
}

Gflash3Vector &GflashShowino::simulateFirstInteractionPoint(int fastSimShowerType, Gflash3Vector &position) {
  // determine the shower starting point (ssp):
  // the position at the entrance + the mean free path till the inelastic
  // interaction inside calo

  double depthAtShower = 0.0;

  // set thePathLengthOnEcal, the pathLength at the reference (r=123.8 for
  // barrel and z=304.5 for endcap)

  // effective interaction length fitter to ssp from Geant4
  double effectiveLambda = 0.0;
  if (theEnergy > 0.0 && theEnergy < 15) {
    effectiveLambda = 24.6 + 2.6 * std::tanh(3.0 * (std::log(theEnergy) - 1.43));
  } else {
    effectiveLambda = 28.4 + 1.20 * std::tanh(1.5 * (std::log(theEnergy) - 4.3));
  }
  // fraction before the crystal, but inside Ecal
  //  double frac_ssp1
  //  = 1.5196e-01+1.3300e-01*tanh(-4.6971e-01*(std::log(theEnergy)+2.4162e+00));
  // fraction after the crystal, but before Hcal
  double frac_ssp2 = 2.8310e+00 + 2.6766e+00 * tanh(-4.8068e-01 * (std::log(theEnergy) + 3.4857e+00));

  if (fastSimShowerType == 100) {  // fastTrack.onEcal() == 1

    //    double rhoTemp = Gflash::ROffCrystalEB +
    //    Gflash::LengthCrystalEB*std::sin(position.getTheta());
    double rhoTemp = Gflash::LengthCrystalEB * std::sin(position.getTheta());
    thePathLengthOnEcal = theHelix->getPathLengthAtRhoEquals(Gflash::RFrontCrystalEB);
    double pathLengthAt2 = theHelix->getPathLengthAtRhoEquals(Gflash::RFrontCrystalEB + rhoTemp);
    double pathLengthAt3 = theHelix->getPathLengthAtRhoEquals(Gflash::Rmin[Gflash::kHB]);

    /// fraction before the crystal, but inside Ecal
    /*
    if(CLHEP::HepUniformRand() < frac_ssp1 ) {
      depthAtShower =
    (pathLengthAt1-thePathLengthOnEcal)*CLHEP::HepUniformRand();
    }
    else {
    */
    // inside the crystal
    //      depthAtShower = (pathLengthAt1-thePathLengthOnEcal) -
    //      effectiveLambda*log(CLHEP::HepUniformRand());
    depthAtShower = -effectiveLambda * log(CLHEP::HepUniformRand());
    // after the crystal
    if (depthAtShower > (pathLengthAt2 - thePathLengthOnEcal)) {
      // before Hcal
      if (CLHEP::HepUniformRand() < frac_ssp2) {
        depthAtShower =
            (pathLengthAt2 - thePathLengthOnEcal) + (pathLengthAt3 - pathLengthAt2) * CLHEP::HepUniformRand();
      }
      // inside Hcal
      else {
        depthAtShower = (pathLengthAt3 - thePathLengthOnEcal) - effectiveLambda * log(CLHEP::HepUniformRand());
        // check whether the shower starts beyond HB
        double pathLengthAt4 = theHelix->getPathLengthAtRhoEquals(Gflash::Rmax[Gflash::kHB]);
        if (depthAtShower > (pathLengthAt4 - thePathLengthOnEcal)) {
          depthAtShower = (pathLengthAt4 - pathLengthAt3) * CLHEP::HepUniformRand();
        }
      }
    }
    //    }
  } else if (fastSimShowerType == 101) {  // fastTrack.onEcal() == 2

    double zTemp = Gflash::LengthCrystalEE;
    double zsign = (position.getEta() > 0) ? 1.0 : -1.0;

    thePathLengthOnEcal = theHelix->getPathLengthAtZ(zsign * Gflash::ZFrontCrystalEE);
    double pathLengthAt2 = theHelix->getPathLengthAtZ(zsign * (Gflash::ZFrontCrystalEE + zTemp));
    double pathLengthAt3 = theHelix->getPathLengthAtZ(zsign * Gflash::Zmin[Gflash::kHE]);

    /*
    if(CLHEP::HepUniformRand() <  frac_ssp1 ) {
      depthAtShower =
    (pathLengthAt1-thePathLengthOnEcal)*CLHEP::HepUniformRand();
    }
    else {
    */
    //      depthAtShower =
    //      (pathLengthAt1-thePathLengthOnEcal)-effectiveLambda*std::log(CLHEP::HepUniformRand());
    depthAtShower = -effectiveLambda * std::log(CLHEP::HepUniformRand());

    if (depthAtShower > (pathLengthAt2 - thePathLengthOnEcal)) {
      if (CLHEP::HepUniformRand() < frac_ssp2) {
        depthAtShower =
            (pathLengthAt2 - thePathLengthOnEcal) + (pathLengthAt3 - pathLengthAt2) * CLHEP::HepUniformRand();
      } else {
        depthAtShower = (pathLengthAt3 - thePathLengthOnEcal) - effectiveLambda * std::log(CLHEP::HepUniformRand());
        // shower starts beyond HE
        double pathLengthAt4 = theHelix->getPathLengthAtZ(zsign * Gflash::Zmax[Gflash::kHE]);
        if (depthAtShower > (pathLengthAt4 - thePathLengthOnEcal)) {
          depthAtShower = (pathLengthAt4 - pathLengthAt3) * CLHEP::HepUniformRand();
        }
      }
    }
    //    }
  } else {
    depthAtShower = 0.0;
  }

  double pathLength = thePathLengthOnEcal + depthAtShower;

  theHelix->getGflashTrajectoryPoint(theTrajectoryPoint, pathLength);

  // return the initial showino position at the shower starting position
  return theTrajectoryPoint.getPosition();
}
