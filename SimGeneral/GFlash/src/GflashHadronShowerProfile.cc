#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimGeneral/GFlash/interface/GflashHadronShowerProfile.h"
#include "SimGeneral/GFlash/interface/GflashTrajectoryPoint.h"
#include "SimGeneral/GFlash/interface/GflashHit.h"

#include <CLHEP/GenericFunctions/IncompleteGamma.hh>
#include <CLHEP/GenericFunctions/LogGamma.hh>
#include <CLHEP/Units/PhysicalConstants.h>
#include <CLHEP/Units/SystemOfUnits.h>
#include <CLHEP/Random/Randomize.h>
#include <CLHEP/Random/RandGaussQ.h>
#include <CLHEP/Random/RandPoissonQ.h>

#include <math.h>

GflashHadronShowerProfile::GflashHadronShowerProfile(edm::ParameterSet parSet) : theParSet(parSet)
{
  theBField = parSet.getParameter<double>("bField");
  theGflashHcalOuter = parSet.getParameter<bool>("GflashHcalOuter");

  theShowino = new GflashShowino();
  theHisto = GflashHistogram::instance();
}

GflashHadronShowerProfile::~GflashHadronShowerProfile() 
{
  if(theShowino) delete theShowino;
}

void GflashHadronShowerProfile::initialize(int showerType, double energy, double globalTime,double charge, 
					   Gflash3Vector &position,Gflash3Vector &momentum) { 

  //initialize GflashShowino for this track
  theShowino->initialize(showerType, energy, globalTime, charge, 
			 position, momentum, theBField);

}

void GflashHadronShowerProfile::hadronicParameterization()
{
  // The skeleton of this method is based on the fortran code gfshow.F originally written  
  // by S. Peters and G. Grindhammer (also see NIM A290 (1990) 469-488), but longitudinal
  // parameterizations of hadron showers are significantly modified for the CMS calorimeter  

  // unit convention: energy in [GeV] and length in [cm]
  // intrinsic properties of hadronic showers (lateral shower profile)
  
  // The step size of showino along the helix trajectory in cm unit
  double showerDepthR50 = 0.0;
  bool firstHcalHit = true;

  //initial valuses that will be changed as the shower developes
  double stepLengthLeft = theShowino->getStepLengthToOut();

  double deltaStep = 0.0;
  double showerDepth = 0.0;

  Gflash::CalorimeterNumber whichCalor = Gflash::kNULL;

  theGflashHitList.clear();

  GflashHit aHit;

  while(stepLengthLeft > 0.0) {

    // update shower depth and stepLengthLeft
    if ( stepLengthLeft < Gflash::divisionStep ) {
      deltaStep = stepLengthLeft;
      stepLengthLeft  = 0.0;
    }
    else {
      deltaStep = Gflash::divisionStep;
      stepLengthLeft -= deltaStep;
    }

    showerDepth += deltaStep;
    showerDepthR50 += deltaStep;

    //update GflashShowino
    theShowino->updateShowino(deltaStep);

    //evaluate energy in this deltaStep along the longitudinal shower profile
    double heightProfile = 0.; 
    double deltaEnergy = 0.;

    whichCalor = Gflash::getCalorimeterNumber(theShowino->getPosition());

    //skip if Showino is outside envelopes
    if(whichCalor == Gflash::kNULL ) continue;

    heightProfile = longitudinalProfile();

    //skip if the delta energy for this step will be very small
    if(heightProfile < 1.00e-08 ) continue;

    //get energy deposition for this step 
    deltaEnergy =  heightProfile*Gflash::divisionStep*energyScale[whichCalor];    
    theShowino->addEnergyDeposited(deltaEnergy);

    //apply sampling fluctuation if showino is inside the sampling calorimeter
    double hadronicFraction = 1.0;
    double fluctuatedEnergy = deltaEnergy;

    int nSpotsInStep = std::max(1,static_cast<int>(getNumberOfSpots(whichCalor)*(deltaEnergy/energyScale[whichCalor]) ));
    double sampleSpotEnergy = hadronicFraction*fluctuatedEnergy/nSpotsInStep;

    // Lateral shape and fluctuations

    if((whichCalor==Gflash::kHB || whichCalor==Gflash::kHE) && firstHcalHit) {
      firstHcalHit = false;
      //reset the showerDepth used in the lateral parameterization inside Hcal
      showerDepthR50 = Gflash::divisionStep;
    }

    //evaluate the fluctuated median of the lateral distribution, R50
    double R50 = medianLateralArm(showerDepthR50,whichCalor);

    double hitEnergy = sampleSpotEnergy*CLHEP::GeV;
    double hitTime = theShowino->getGlobalTime()*CLHEP::nanosecond;

    Gflash::CalorimeterNumber hitCalor = Gflash::kNULL;

    for (int ispot = 0 ;  ispot < nSpotsInStep ; ispot++) {
	
      // Compute global position of generated spots with taking into account magnetic field
      // Divide deltaStep into nSpotsInStep and give a spot a global position
      double incrementPath = theShowino->getPathLength() 
	+ (deltaStep/nSpotsInStep)*(ispot+0.5 - 0.5*nSpotsInStep);
	
      // trajectoryPoint give a spot an imaginary point along the shower development
      GflashTrajectoryPoint trajectoryPoint;
      theShowino->getHelix()->getGflashTrajectoryPoint(trajectoryPoint,incrementPath);
	
      Gflash3Vector hitPosition = locateHitPosition(trajectoryPoint,R50);   
	
      hitCalor = Gflash::getCalorimeterNumber(hitPosition);	  

      if( hitCalor == Gflash::kNULL) continue;

      hitPosition *= CLHEP::cm;

      aHit.setTime(hitTime);
      aHit.setEnergy(hitEnergy);
      aHit.setPosition(hitPosition);
      theGflashHitList.push_back(aHit);
	
    } // end of for spot iteration
  } // end of while for longitudinal integration

  //HO parameterization

  if(theShowino->getEnergy()> Gflash::MinEnergyCutOffForHO &&
     fabs(theShowino->getPositionAtShower().pseudoRapidity()) < Gflash::EtaMax[Gflash::kHO] && 
     theGflashHcalOuter) {
    
    //non zero ho fraction to simulate based on geant4
    double nonzeroProb = 0.7*fTanh(theShowino->getEnergy(),Gflash::ho_nonzero);
    double r0 = CLHEP::HepUniformRand();
    double leftoverE =  theShowino->getEnergy() - theShowino->getEnergyDeposited();

    //@@@ nonzeroProb is not random - need further correlation for non-zero HO energy

    if( r0 < nonzeroProb && leftoverE > 0.0) {
            
      //starting path Length and stepLengthLeft
      double pathLength = theShowino->getHelix()->getPathLengthAtRhoEquals(Gflash::Rmin[Gflash::kHO]-10);
      stepLengthLeft = theShowino->getHelix()->getPathLengthAtRhoEquals(Gflash::Rmax[Gflash::kHO]+10) - pathLength;
      showerDepth = pathLength - theShowino->getPathLengthAtShower();
      
      theShowino->setPathLength(pathLength);
      
      double pathLengthx = theShowino->getHelix()->getPathLengthAtRhoEquals(Gflash::Rmax[Gflash::kHO]);
      double pathLengthy = theShowino->getHelix()->getPathLengthAtRhoEquals(Gflash::Rmax[Gflash::kHB]);
      
      while(stepLengthLeft > 0.0) {
	
	// update shower depth and stepLengthLeft
	if ( stepLengthLeft < Gflash::divisionStep ) {
	  deltaStep = stepLengthLeft;
	  stepLengthLeft  = 0.0;
	}
	else {
	  deltaStep = Gflash::divisionStep;
	  stepLengthLeft -= deltaStep;
	}
	
	showerDepth += deltaStep;
	
	//update GflashShowino
	theShowino->updateShowino(deltaStep);
	
	//evaluate energy in this deltaStep along the longitudinal shower profile
	double heightProfile = 0.; 
	double deltaEnergy = 0.;
	
	double hoScale  = leftoverE*(pathLengthx-pathLengthy)/(pathLengthx- theShowino->getPathLengthAtShower());
	double refDepth = theShowino->getPathLength() 
	  - theShowino->getHelix()->getPathLengthAtRhoEquals(Gflash::Rmax[Gflash::kHB]);
	
	if( refDepth > 0) {
	  heightProfile = hoProfile(theShowino->getPathLength(),refDepth);
	  deltaEnergy = heightProfile*Gflash::divisionStep*hoScale;
	}
	
	int nSpotsInStep = std::max(50,static_cast<int>((160.+40* CLHEP::RandGaussQ::shoot())*std::log(theShowino->getEnergy())+50.));
	
      	double hoFraction = 1.00;
	double poissonProb = CLHEP::RandPoissonQ::shoot(1.0);
	
	double fluctuatedEnergy = deltaEnergy*poissonProb;
	double sampleSpotEnergy = hoFraction*fluctuatedEnergy/nSpotsInStep;
	
	// Lateral shape and fluctuations
	
	//evaluate the fluctuated median of the lateral distribution, R50
	double R50 = medianLateralArm(showerDepth,Gflash::kHB);

	double hitEnergy = sampleSpotEnergy*CLHEP::GeV;
	double hitTime = theShowino->getGlobalTime()*CLHEP::nanosecond;
	
	for (int ispot = 0 ;  ispot < nSpotsInStep ; ispot++) {
	  
	  double incrementPath = theShowino->getPathLength() 
	    + (deltaStep/nSpotsInStep)*(ispot+0.5 - 0.5*nSpotsInStep);
	  
	  // trajectoryPoint give a spot an imaginary point along the shower development
	  GflashTrajectoryPoint trajectoryPoint;
	  theShowino->getHelix()->getGflashTrajectoryPoint(trajectoryPoint,incrementPath);
	  
	  Gflash3Vector hitPosition = locateHitPosition(trajectoryPoint,R50);   
	  hitPosition *= CLHEP::cm;

	  if(std::fabs(hitPosition.getZ()/CLHEP::cm) > Gflash::Zmax[Gflash::kHO] ) continue;

	  aHit.setTime(hitTime);
	  aHit.setEnergy(hitEnergy);
	  aHit.setPosition(hitPosition);
	  theGflashHitList.push_back(aHit);
	  
	} // end of for HO spot iteration
      } // end of while for HO longitudinal integration
    }
  }

  //  delete theGflashNavigator;

}

void GflashHadronShowerProfile::loadParameters()
{
  edm::LogInfo("SimGeneralGFlash") << "GflashHadronShowerProfile::loadParameters() "
				   << "should be implimented for each particle type";
}

double GflashHadronShowerProfile::medianLateralArm(double showerDepthR50, Gflash::CalorimeterNumber kCalor) 
{
  double lateralArm = 0.0;
  if(kCalor != Gflash::kNULL) {
    
    double showerDepthR50X = std::min(showerDepthR50/22.4, Gflash::maxShowerDepthforR50);
    double R50          = lateralPar[kCalor][0] + std::max(0.0,lateralPar[kCalor][1]) * showerDepthR50X;
    double varinanceR50 = std::pow((lateralPar[kCalor][2] + lateralPar[kCalor][3] * showerDepthR50X) * R50, 2);
    
    // Simulation of lognormal distribution
    
    if(R50>0) {
      double sigmaSq  = std::log(varinanceR50/(R50*R50)+1.0);
      double sigmaR50 = std::sqrt(sigmaSq);
      double meanR50  = std::log(R50) - (sigmaSq/2.);
      
      lateralArm = std::exp(meanR50 + sigmaR50* CLHEP::RandGaussQ::shoot());
    }
  }
  return lateralArm;
}

Gflash3Vector GflashHadronShowerProfile::locateHitPosition(GflashTrajectoryPoint& point, double lateralArm) 
{
  // Smearing in r according to f(r)= 2.*r*R50**2/(r**2+R50**2)**2
  double rnunif = CLHEP::HepUniformRand();
  double rxPDF = std::sqrt(rnunif/(1.-rnunif));
  double rShower = lateralArm*rxPDF;
  
  //rShower within maxLateralArmforR50
  rShower = std::min(Gflash::maxLateralArmforR50,rShower);

  // Uniform smearing in phi
  double azimuthalAngle = CLHEP::twopi*CLHEP::HepUniformRand(); 

  // actual spot position by adding a radial vector to a trajectoryPoint
  Gflash3Vector position = point.getPosition() +
    rShower*std::cos(azimuthalAngle)*point.getOrthogonalUnitVector() +
    rShower*std::sin(azimuthalAngle)*point.getCrossUnitVector();
  
  //@@@debugging histograms
  if(theHisto->getStoreFlag()) {
    theHisto->rshower->Fill(rShower);
    theHisto->lateralx->Fill(rShower*std::cos(azimuthalAngle));
    theHisto->lateraly->Fill(rShower*std::sin(azimuthalAngle));
  }
  return position;
}

//double GflashHadronShowerProfile::longitudinalProfile(double showerDepth, double pathLength) {
double GflashHadronShowerProfile::longitudinalProfile() {

  double heightProfile = 0.0;

  Gflash3Vector pos = theShowino->getPosition();
  int showerType = theShowino->getShowerType();
  double showerDepth = theShowino->getDepth();
  double transDepth = theShowino->getStepLengthToHcal();

  // Energy in a delta step (dz) = (energy to deposite)*[Gamma(z+dz)-Gamma(z)]*dz
  // where the incomplete Gamma function gives an intergrate probability of the longitudinal 
  // shower up to the shower depth (z).
  // Instead, we use approximated energy; energy in dz = (energy to deposite)*gamma(z)*dz
  // where gamma is the Gamma-distributed probability function

  Gflash::CalorimeterNumber whichCalor = Gflash::getCalorimeterNumber(pos);

  if(showerType == 0 || showerType == 4 ) {
    double shiftDepth = theShowino->getPathLengthOnEcal() - theShowino->getPathLengthAtShower();
    if(shiftDepth > 0 ) {
      heightProfile = twoGammaProfile(longEcal,showerDepth-shiftDepth,whichCalor);
    }
    else  {
      heightProfile = 0.;
      //      std::cout << "negative shiftDepth for showerType 0 " << shiftDepth << std::endl;
    }
  }  
  else if(showerType == 1 || showerType == 5 ) {
    if(whichCalor == Gflash::kESPM || whichCalor == Gflash::kENCA ) {
      heightProfile = twoGammaProfile(longEcal,showerDepth,whichCalor);
    }
    else if(whichCalor == Gflash::kHB || whichCalor == Gflash::kHE) {
      heightProfile = twoGammaProfile(longHcal,showerDepth-transDepth,whichCalor);
    }
    else  heightProfile = 0.;
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

double GflashHadronShowerProfile::hoProfile(double pathLength, double refDepth) {

  double heightProfile = 0;

  GflashTrajectoryPoint tempPoint;
  theShowino->getHelix()->getGflashTrajectoryPoint(tempPoint,pathLength);

  double dint = 1.4*Gflash::intLength[Gflash::kHO]*std::sin(tempPoint.getPosition().getTheta());
  heightProfile = std::exp(-1.0*refDepth/dint);

  return heightProfile;
}

void GflashHadronShowerProfile::getFluctuationVector(double *lowTriangle, double *correlationVector) {

    const int dim = Gflash::NPar;

    double **xr   = new double *[dim];
    double **xrho = new double *[dim];
    
    for(int j=0;j<dim;j++) {
      xr[j]   = new double [dim];
      xrho[j] = new double [dim];
    }
    
    for(int i = 0; i < dim; i++) {
      for(int j = 0; j < i+1 ; j++) {
        if(j==i) xrho[i][j] = 1.0;
	else {
	  xrho[i][j] = lowTriangle[i*(i-1)/2 + j];
	  xrho[j][i] = xrho[i][j];
	}
      }
    }

    doCholeskyReduction(xrho,xr,dim);

    for(int i = 0 ; i < dim ; i++) {
      for (int j = 0 ; j < i+1 ; j++){
	correlationVector[i*(i+1)/2 + j] = xr[i][j];
      }
    }

    for(int j=0;j<dim;j++) delete [] xr[j];
    delete [] xr;
    for(int j=0;j<dim;j++) delete [] xrho[j];
    delete [] xrho;
}

void GflashHadronShowerProfile::doCholeskyReduction(double **vv, double **cc, const int ndim) {

  double sumCjkSquare;
  double vjjLess;
  double sumCikjk;

  cc[0][0] = std::sqrt(vv[0][0]);

  for(int j=1 ; j < ndim ; j++) {
    cc[j][0] = vv[j][0]/cc[0][0];
  }

  for(int j=1 ; j < ndim ; j++) {

    sumCjkSquare = 0.0;
    for (int k=0 ; k < j ; k++) sumCjkSquare += cc[j][k]*cc[j][k];

    vjjLess =  vv[j][j] - sumCjkSquare;

    //check for the case that vjjLess is negative
    cc[j][j] = std::sqrt(std::fabs(vjjLess));

    for (int i=j+1 ; i < ndim ; i++) {
      sumCikjk = 0.;
      for(int k=0 ; k < j ; k++) sumCikjk += cc[i][k]*cc[j][k];
      cc[i][j] = (vv[i][j] - sumCikjk)/cc[j][j];
    }
  }
}

int GflashHadronShowerProfile::getNumberOfSpots(Gflash::CalorimeterNumber kCalor) {
  //generator number of spots: energy dependent Gamma distribution of Nspots based on Geant4
  //replacing old parameterization of H1,
  //int numberOfSpots = std::max( 50, static_cast<int>(80.*std::log(einc)+50.));

  double einc = theShowino->getEnergy();
  int showerType = theShowino->getShowerType();

  int numberOfSpots = 0;
  double nmean  = 0.0;
  double nsigma = 0.0;

  if(showerType == 0 || showerType == 1 || showerType == 4 || showerType == 5 ) {
    if(kCalor == Gflash::kESPM || kCalor == Gflash::kENCA) {
      nmean = 10000 + 5000*log(einc);
      nsigma = 1000;
    }
    if(kCalor == Gflash::kHB || kCalor == Gflash::kHE) {
      nmean =  5000 + 2500*log(einc);
      nsigma =  500;
    }
  }
  else if (showerType == 2 || showerType == 3 || showerType == 6 || showerType == 7 ) {
    if(kCalor == Gflash::kHB || kCalor == Gflash::kHE) {
      nmean =  5000 + 2500*log(einc);
      nsigma =  500;
    }
    else {
      nmean = 10000;
      nsigma = 1000;
    }
  }
  //@@@need correlation and individual fluctuation on alphaNspots and betaNspots here:
  //evaluating covariance should be straight forward since the distribution is 'one' Gamma

  numberOfSpots = std::max(500,static_cast<int> (nmean+nsigma* CLHEP::RandGaussQ::shoot()));

  //until we optimize the reduction scale in the number of Nspots
      
  if( kCalor == Gflash::kESPM ||  kCalor == Gflash::kENCA) {
    numberOfSpots = static_cast<int>(numberOfSpots/100);
  }
  else {
    numberOfSpots = static_cast<int>(numberOfSpots/3.0);
  }

  return numberOfSpots;
}

double GflashHadronShowerProfile::fTanh(double einc, const double *par) {
  double func = 0.0;
  if(einc>0.0) func = par[0]+par[1]*std::tanh(par[2]*(std::log(einc)-par[3])) + par[4]*std::log(einc);
  return func;
}

double GflashHadronShowerProfile::fLnE1(double einc, const double *par) {
  double func = 0.0;
  if(einc>0.0) func = par[0]+par[1]*std::log(einc);
  return func;
}

double GflashHadronShowerProfile::depthScale(double ssp, double ssp0, double length) {
  double func = 0.0;
  if(length>0.0) func = std::pow((ssp-ssp0)/length,2.0);
  return func;
}

double GflashHadronShowerProfile::gammaProfile(double alpha, double beta, double showerDepth, double lengthUnit) {
  double gamma = 0.0;
  //  if(alpha > 0 && beta > 0 && lengthUnit > 0) {
  if(showerDepth>0.0) {
    Genfun::LogGamma lgam;
    double x = showerDepth*(beta/lengthUnit);
    gamma = (beta/lengthUnit)*std::pow(x,alpha-1.0)*std::exp(-x)/std::exp(lgam(alpha));
  }
  return gamma;
}
 
double GflashHadronShowerProfile::twoGammaProfile(double *longPar, double depth, Gflash::CalorimeterNumber kIndex) {
  double twoGamma = 0.0;

  longPar[0] = std::min(1.0,longPar[0]);
  longPar[0] = std::max(0.0,longPar[0]);

  if(longPar[3] > 4.0 || longPar[4] > 4.0) {
    double rfactor = 2.0/std::max(longPar[3],longPar[4]);
    longPar[3] = rfactor*(longPar[3]+1.0);  
    longPar[4] *= rfactor;  
  }

  twoGamma  = longPar[0]*gammaProfile(exp(longPar[1]),exp(longPar[2]),depth,Gflash::radLength[kIndex])
         +(1-longPar[0])*gammaProfile(exp(longPar[3]),exp(longPar[4]),depth,Gflash::intLength[kIndex]);
  return twoGamma;
}
