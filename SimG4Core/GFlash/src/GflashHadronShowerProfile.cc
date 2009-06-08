#include "SimG4Core/GFlash/interface/GflashHadronShowerProfile.h"
#include "SimG4Core/GFlash/interface/GflashTrajectoryPoint.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/GenericFunctions/IncompleteGamma.hh"
#include "CLHEP/GenericFunctions/LogGamma.hh"
#include "Randomize.hh"
#include "G4TransportationManager.hh"
#include "G4TouchableHandle.hh"
#include "G4VSensitiveDetector.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Proton.hh"

#include "G4AntiProton.hh"
#include "G4VProcess.hh"

#include <math.h>

GflashHadronShowerProfile::GflashHadronShowerProfile(edm::ParameterSet parSet) : theParSet(parSet)
{
  theBField = parSet.getParameter<double>("bField");
  theGflashHcalOuter = parSet.getParameter<bool>("GflashHcalOuter");

  theShowerType   = -1;

  theGflashStep = new G4Step();
  theShowino = new GflashShowino();
  theGflashTouchableHandle = new G4TouchableHistory();

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
  theRandPoissonQ = new CLHEP::RandPoissonQ(rng->getEngine());
  theRandChiSquare = new CLHEP::RandChiSquare(rng->getEngine());
}

GflashHadronShowerProfile::~GflashHadronShowerProfile() 
{
  if(theGflashStep) delete theGflashStep;
  if(theShowino) delete theShowino;
  if(theRandGauss) delete theRandGauss;
  if(theRandGamma) delete theRandGamma;
  if(theRandPoissonQ) delete theRandPoissonQ;
}


void GflashHadronShowerProfile::hadronicParameterization(const G4FastTrack& fastTrack)
{
  // The skeleton of this method is based on the fortran code gfshow.F originally written  
  // by S. Peters and G. Grindhammer (also see NIM A290 (1990) 469-488), but longitudinal
  // parameterizations of hadron showers are significantly modified for the CMS calorimeter  

  // unit convention: energy in [GeV] and length in [cm]
  // intrinsic properties of hadronic showers (lateral shower profile)
  
  //initialize GflashShowino for this track
  theShowino->initializeShowino(fastTrack,theBField);
  
  // The step size of showino along the helix trajectory in cm unit
  G4double showerDepthR50 = 0.0;
  bool firstHcalHit = true;

  //set track for the Gflash Step
  theGflashStep->SetTrack(const_cast<G4Track*>(fastTrack.GetPrimaryTrack()));

  // navigator and time information is needed for making a fake step

  theGflashNavigator = new G4Navigator();
  theGflashNavigator->SetWorldVolume(G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume());

  //initial valuses that will be changed as the shower developes
  G4double stepLengthLeft = theShowino->getStepLengthToOut();
  Gflash::CalorimeterNumber whichCalor =  Gflash::getCalorimeterNumber(theShowino->getPosition());

  G4double deltaStep = 0.0;
  G4double showerDepth = 0.0;

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

    //get energy deposition for this step 
    whichCalor = Gflash::getCalorimeterNumber(theShowino->getPosition());
    heightProfile = longitudinalProfile(showerDepth,theShowino->getPathLength());
    deltaEnergy =  heightProfile*Gflash::divisionStep*energyScale[whichCalor];    

    theShowino->addEnergyDeposited(deltaEnergy);

    //@@@ When depthShower is inside Hcal, the sampling fluctuation for deposited
    //    energy will be treated in SD.  However we should put some scale factor 
    //    to relate the spot energy to the energy deposited in each geant4 step. 

    double hadronicFraction = 1.0;

    G4double fluctuatedEnergy = deltaEnergy;

    //apply sampling fluctuation if showino is inside the sampling calorimeter
    //according to averageSpotEnergy = c*c*(energyToDeposit/einc) where
    //sigma/energyToDeposit = c/sqrt(einc)

    G4int nSpotsInStep = std::max(1,static_cast<int>(getNumberOfSpots(whichCalor)*(deltaEnergy/energyScale[whichCalor]) ));
    G4double sampleSpotEnergy = hadronicFraction*fluctuatedEnergy/nSpotsInStep;

    // Lateral shape and fluctuations

    if((whichCalor==Gflash::kHB || whichCalor==Gflash::kHE) && firstHcalHit) {
      firstHcalHit = false;
      //reset the showerDepth used in the lateral parameterization inside Hcal
      showerDepthR50 = Gflash::divisionStep;
    }

    //evaluate the fluctuated median of the lateral distribution, R50
    G4double R50 = medianLateralArm(showerDepthR50,whichCalor);

    for (G4int ispot = 0 ;  ispot < nSpotsInStep ; ispot++) {
   
      // Compute global position of generated spots with taking into account magnetic field
      // Divide deltaStep into nSpotsInStep and give a spot a global position
      G4double incrementPath = theShowino->getPathLength() 
	                     + (deltaStep/nSpotsInStep)*(ispot+0.5 - 0.5*nSpotsInStep);

      // trajectoryPoint give a spot an imaginary point along the shower development
      GflashTrajectoryPoint trajectoryPoint;
      theShowino->getHelix()->getGflashTrajectoryPoint(trajectoryPoint,incrementPath);

      G4ThreeVector spotPosition = locateSpotPosition(trajectoryPoint,R50);   

      spotPosition *= cm;
      G4double spotEnergy = sampleSpotEnergy*GeV;

      //@@@ temporary hit time for each fake step - need to be implemented correctly
      G4double timeGlobal = theShowino->getGlobalTime() + 0.0001*ispot*nanosecond;

      // fill equivalent changes to a (fake) step associated with a spot 

      theGflashNavigator->LocateGlobalPointAndUpdateTouchableHandle(spotPosition,G4ThreeVector(0,0,0),theGflashTouchableHandle, false);
      updateGflashStep(spotPosition,timeGlobal);

      // Send G4Step information to Hit/Dig if the volume is sensitive

      G4VPhysicalVolume* aCurrentVolume = theGflashStep->GetPreStepPoint()->GetPhysicalVolume();
      if( aCurrentVolume == 0 ) continue;

      G4LogicalVolume* lv = aCurrentVolume->GetLogicalVolume();
      if(lv->GetRegion()->GetName() != "CaloRegion") continue;

      theGflashStep->GetPreStepPoint()->SetSensitiveDetector(aCurrentVolume->GetLogicalVolume()->GetSensitiveDetector());

      G4VSensitiveDetector* aSensitive = theGflashStep->GetPreStepPoint()->GetSensitiveDetector();

      if( aSensitive == 0 || (std::fabs(spotPosition.getZ()/cm) > Gflash::Zmax[Gflash::kHE]) ) continue;

      G4String nameCalor = aCurrentVolume->GetName();
      nameCalor.assign(nameCalor,0,2);
      if(nameCalor == "HB" || nameCalor=="HE") spotEnergy *= Gflash::scaleSensitive;

      theGflashStep->SetTotalEnergyDeposit(spotEnergy);
      theGflashStep->GetPostStepPoint()->SetProcessDefinedStep(const_cast<G4VProcess*> 
		    (fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()));

      aSensitive->Hit(theGflashStep);

    } // end of for spot iteration
  } // end of while for longitudinal integration

  //HO parameterization

  if(theShowino->getEnergy()> Gflash::MinEnergyCutOffForHO &&
     fabs(theShowino->getPositionAtShower().pseudoRapidity()) < Gflash::EtaMax[Gflash::kHO] && 
     theGflashHcalOuter) {
    
    //non zero ho fraction to simulate based on geant4
    G4double nonzeroProb = 0.7*fTanh(theShowino->getEnergy(),Gflash::ho_nonzero);
    G4double r0 = G4UniformRand();
    G4double leftoverE =  theShowino->getEnergy() - theShowino->getEnergyDeposited();

    //@@@ nonzeroProb is not random - need further correlation for non-zero HO energy

    if( r0 < nonzeroProb && leftoverE > 0.0) {
            
      //starting path Length and stepLengthLeft
      G4double pathLength = theShowino->getHelix()->getPathLengthAtRhoEquals(Gflash::Rmin[Gflash::kHO]-10);
      stepLengthLeft = theShowino->getHelix()->getPathLengthAtRhoEquals(Gflash::Rmax[Gflash::kHO]+10) - pathLength;
      showerDepth = pathLength - theShowino->getPathLengthAtShower();
      
      theShowino->setPathLength(pathLength);
      
      G4double pathLengthx = theShowino->getHelix()->getPathLengthAtRhoEquals(Gflash::Rmax[Gflash::kHO]);
      G4double pathLengthy = theShowino->getHelix()->getPathLengthAtRhoEquals(Gflash::Rmax[Gflash::kHB]);
      
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
	
	G4double hoScale  = leftoverE*(pathLengthx-pathLengthy)/(pathLengthx- theShowino->getPathLengthAtShower());
	G4double refDepth = theShowino->getPathLength() 
	  - theShowino->getHelix()->getPathLengthAtRhoEquals(Gflash::Rmax[Gflash::kHB]);
	
	if( refDepth > 0) {
	  heightProfile = hoProfile(theShowino->getPathLength(),refDepth);
	  deltaEnergy = heightProfile*Gflash::divisionStep*hoScale;
	}
	
	G4int nSpotsInStep = std::max(50,static_cast<int>((160.+40*theRandGauss->fire())*std::log(theShowino->getEnergy())+50.));
	
      	double hoFraction = 1.00;
	G4double poissonProb = theRandPoissonQ->fire(1.0);
	//@@@ may need an additional rejection here
	//	if(poissonProb > 0.0) {
	
	G4double fluctuatedEnergy = deltaEnergy*poissonProb;
	G4double sampleSpotEnergy = hoFraction*fluctuatedEnergy/nSpotsInStep;
	
	// Lateral shape and fluctuations
	
	//evaluate the fluctuated median of the lateral distribution, R50
	G4double R50 = medianLateralArm(showerDepth,Gflash::kHB);
	
	for (G4int ispot = 0 ;  ispot < nSpotsInStep ; ispot++) {
	  
	  G4double incrementPath = theShowino->getPathLength() 
	    + (deltaStep/nSpotsInStep)*(ispot+0.5 - 0.5*nSpotsInStep);
	  
	  // trajectoryPoint give a spot an imaginary point along the shower development
	  GflashTrajectoryPoint trajectoryPoint;
	  theShowino->getHelix()->getGflashTrajectoryPoint(trajectoryPoint,incrementPath);
	  
	  G4ThreeVector spotPosition = locateSpotPosition(trajectoryPoint,R50);   
	  
	  spotPosition *= cm;
	  G4double spotEnergy = sampleSpotEnergy*GeV;
	  
	  //@@@ temporary hit time for each fake step - need to be implemented correctly
	  G4double timeGlobal = theShowino->getGlobalTime() + 0.0001*ispot*nanosecond ;
	  
	  // fill equivalent changes to a (fake) step associated with a spot 
	  
	  theGflashNavigator->LocateGlobalPointAndUpdateTouchableHandle(spotPosition,G4ThreeVector(0,0,0),theGflashTouchableHandle, false);
	  updateGflashStep(spotPosition,timeGlobal);
	  
	  // Send G4Step information to Hit/Dig if the volume is sensitive
	  
	  G4VPhysicalVolume* aCurrentVolume = theGflashStep->GetPreStepPoint()->GetPhysicalVolume();
	  if( aCurrentVolume == 0 ) continue;
	  
	  G4LogicalVolume* lv = aCurrentVolume->GetLogicalVolume();
	  if(lv->GetRegion()->GetName() != "CaloRegion") continue;
	  
	  theGflashStep->GetPreStepPoint()->SetSensitiveDetector(aCurrentVolume->GetLogicalVolume()->GetSensitiveDetector());
	  
	  G4VSensitiveDetector* aSensitive = theGflashStep->GetPreStepPoint()->GetSensitiveDetector();
	  
	  if( aSensitive == 0 || (std::fabs(spotPosition.getZ()/cm) > Gflash::Zmax[Gflash::kHE]) ) continue;
	  
	  G4String nameCalor = aCurrentVolume->GetName();
	  nameCalor.assign(nameCalor,0,2);
	  if(nameCalor == "HT") spotEnergy *= Gflash::scaleSensitive;
	  
	  theGflashStep->SetTotalEnergyDeposit(spotEnergy);
	  theGflashStep->GetPostStepPoint()->SetProcessDefinedStep(const_cast<G4VProcess*> 
		        (fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()));
	  
	  aSensitive->Hit(theGflashStep);
	  
	} // end of for HO spot iteration
	  //	}
      } // end of while for HO longitudinal integration
    }
  }

  delete theGflashNavigator;

}

void GflashHadronShowerProfile::loadParameters(const G4FastTrack& fastTrack)
{
  std::cout << "GflashHadronShowerProfile::loadParameters should be implimented for "
	    << fastTrack.GetPrimaryTrack()->GetDefinition()->GetParticleName()
	    << std::endl;
}

void GflashHadronShowerProfile::updateGflashStep(G4ThreeVector spotPosition, G4double timeGlobal)
{
  theGflashStep->GetPostStepPoint()->SetGlobalTime(timeGlobal);
  theGflashStep->GetPreStepPoint()->SetPosition(spotPosition);
  theGflashStep->GetPostStepPoint()->SetPosition(spotPosition);
  theGflashStep->GetPreStepPoint()->SetTouchableHandle(theGflashTouchableHandle);
}

G4double GflashHadronShowerProfile::medianLateralArm(G4double showerDepthR50, Gflash::CalorimeterNumber kCalor) 
{
  G4double lateralArm = 0.0;
  if(kCalor != Gflash::kNULL) {
    
    double showerDepthR50X = std::min(showerDepthR50/22.4, Gflash::maxShowerDepthforR50);
    double R50          = lateralPar[kCalor][0] + std::max(0.0,lateralPar[kCalor][1]) * showerDepthR50X;
    double varinanceR50 = std::pow((lateralPar[kCalor][2] + lateralPar[kCalor][3] * showerDepthR50X) * R50, 2);
    
    // Simulation of lognormal distribution
    
    double sigmaSq  = std::log(varinanceR50/(R50*R50)+1.0);
    double sigmaR50 = std::sqrt(sigmaSq);
    double meanR50  = std::log(R50) - (sigmaSq/2.);
    
    lateralArm = std::exp(meanR50 + sigmaR50*theRandGauss->fire());
  }
  return lateralArm;
}

G4ThreeVector GflashHadronShowerProfile::locateSpotPosition(GflashTrajectoryPoint& point, G4double lateralArm) 
{
  // Smearing in r according to f(r)= 2.*r*R50**2/(r**2+R50**2)**2
  G4double rnunif = G4UniformRand();
  G4double rxPDF = std::sqrt(rnunif/(1.-rnunif));
  G4double rShower = lateralArm*rxPDF;
  
  // Uniform smearing in phi, for 66% of lateral containm.
  G4double azimuthalAngle = twopi*G4UniformRand(); 

  // actual spot position by adding a radial vector to a trajectoryPoint
  G4ThreeVector position = point.getPosition() +
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
void GflashHadronShowerProfile::setShowerType(const G4FastTrack& fastTrack)
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
	if( posRho < 129.0 ) showerType = 0;
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
	if(fabs(position.getZ()) < 330.0 ) showerType = 4;
      }
      else showerType = 7;
    }
    //@@@need z-dependent correction on the mean energy reponse
  }

  theShowerType = showerType;
}

G4double GflashHadronShowerProfile::longitudinalProfile(G4double showerDepth, G4double pathLength) {

  G4double heightProfile = 0;
  G4double transDepth = theShowino->getStepLengthToHcal();
  G4ThreeVector pos = theShowino->getPosition();
  //  G4double einc = theShowino->getEnergy();

  // Energy in a delta step (dz) = (energy to deposite)*[Gamma(z+dz)-Gamma(z)]*dz
  // where the incomplete Gamma function gives an intergrate probability of the longitudinal 
  // shower up to the shower depth (z).
  // Instead, we use approximated energy; energy in dz = (energy to deposite)*gamma(z)*dz
  // where gamma is the Gamma-distributed probability function

  GflashTrajectoryPoint tempPoint;
  theShowino->getHelix()->getGflashTrajectoryPoint(tempPoint,pathLength);

  if(theShowerType == 0 || theShowerType == 1 ) {
    //@@@@change 152 to the 129+22*sin(theta) type of boundary
    if(tempPoint.getPosition().getRho() < 152.0 ) { 
      heightProfile = twoGammaProfile(longEcal,showerDepth,Gflash::kESPM);
    }
    else if (tempPoint.getPosition().getRho() > Gflash::Rmin[Gflash::kHB] ){
      heightProfile = twoGammaProfile(longHcal,showerDepth-transDepth,Gflash::kHB);
    }
    else heightProfile = 0.;
  }  
  else if(theShowerType == 4 || theShowerType == 5){
    //@@@use new parameterization for EE/HE
    if(std::abs(tempPoint.getPosition().getZ()) < Gflash::Zmin[Gflash::kENCA]+23.0 ) { 
      heightProfile = twoGammaProfile(longEcal,showerDepth,Gflash::kENCA);
    }
    else if (std::abs(tempPoint.getPosition().getZ()) > Gflash::Rmin[Gflash::kHE] ){
      heightProfile = twoGammaProfile(longHcal,showerDepth-transDepth,Gflash::kHE);
    }
    else heightProfile = 0.;
  }  
  else if (theShowerType == 2 || theShowerType == 6 ) {
    //two gammas between crystal and Hcal
    if((showerDepth - transDepth) > 0.0) {
      heightProfile = twoGammaProfile(longHcal,showerDepth-transDepth,Gflash::kHB);
    }
    else heightProfile = 0.;
  }
  else if (theShowerType == 3 || theShowerType == 7 ) {
    //two gammas inside Hcal
    heightProfile = twoGammaProfile(longHcal,showerDepth,Gflash::kHB);
  }

  return heightProfile;
}

G4double GflashHadronShowerProfile::hoProfile(G4double pathLength, G4double refDepth) {

  G4double heightProfile = 0;

  GflashTrajectoryPoint tempPoint;
  theShowino->getHelix()->getGflashTrajectoryPoint(tempPoint,pathLength);

  G4double dint = 1.4*Gflash::intLength[Gflash::kHO]*std::sin(tempPoint.getPosition().getTheta());
  heightProfile = std::exp(-1.0*refDepth/dint);

  return heightProfile;
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

    //check for the case that vjjLess is negative
    cc[j][j] = std::sqrt(std::fabs(vjjLess));

    for (G4int i=j+1 ; i < ndim ; i++) {
      sumCikjk = 0.;
      for(G4int k=0 ; k < j ; k++) sumCikjk += cc[i][k]*cc[j][k];
      cc[i][j] = (vv[i][j] - sumCikjk)/cc[j][j];
    }
  }
}

G4int GflashHadronShowerProfile::getNumberOfSpots(Gflash::CalorimeterNumber kCalor) {
  //generator number of spots: energy dependent Gamma distribution of Nspots based on Geant4
  //replacing old parameterization of H1,
  //G4int numberOfSpots = std::max( 50, static_cast<int>(80.*std::log(einc)+50.));

  G4double einc = theShowino->getEnergy();

  G4int numberOfSpots = 0;
  G4double nmean  = 0.0;
  G4double nsigma = 0.0;

  if(theShowerType == 0 || theShowerType == 1 || theShowerType == 4 || theShowerType == 5 ) {
    if(kCalor == Gflash::kESPM) {
      nmean = 10000 + 5000*log(einc);
      nsigma = 1000;
    }
    if(kCalor == Gflash::kHB) {
      nmean =  5000 + 2500*log(einc);
      nsigma =  500;
    }
  }
  else if (theShowerType == 2 || theShowerType == 3 || theShowerType == 6 || theShowerType == 7 ) {
    if(kCalor == Gflash::kHB) {
      nmean =  5000 + 2500*log(einc);
      nsigma =  500;
    }
  }
  else {
      nmean = 10000;
      nsigma = 1000;
  }

  //@@@need correlation and individual fluctuation on alphaNspots and betaNspots here:
  //evaluating covariance should be straight forward since the distribution is 'one' Gamma

  numberOfSpots = std::max(500,static_cast<int> (nmean+nsigma*theRandGauss->fire()));

  //until we optimize the reduction scale in the number of Nspots
      
  if( (theShowino->getPosition()).getRho() < 153.0 ) {
    numberOfSpots = static_cast<int>(numberOfSpots/100);
  }
  else {
    numberOfSpots = static_cast<int>(numberOfSpots/3.0);
  }

  return numberOfSpots;
}

G4double GflashHadronShowerProfile::fTanh(G4double einc, const G4double *par) {
  double func = 0.0;
  if(einc>0.0) func = par[0]+par[1]*std::tanh(par[2]*(std::log(einc)-par[3]));
  return func;
}

G4double GflashHadronShowerProfile::fLnE1(G4double einc, const G4double *par) {
  double func = 0.0;
  if(einc>0.0) func = par[0]+par[1]*std::log(einc);
  return func;
}

G4double GflashHadronShowerProfile::depthScale(G4double ssp, G4double ssp0, G4double length) {
  double func = 0.0;
  if(length>0.0) func = std::pow((ssp-ssp0)/length,2.0);
  return func;
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
 
G4double GflashHadronShowerProfile::twoGammaProfile(G4double *longPar, G4double depth, Gflash::CalorimeterNumber kIndex) {
  G4double twoGamma = 0.0;
  twoGamma  = longPar[0]* gammaProfile(exp(longPar[1]),exp(longPar[2]),depth,Gflash::radLength[kIndex])
          +(1-longPar[0])*gammaProfile(exp(longPar[3]),exp(longPar[4]),depth,Gflash::intLength[kIndex]);
  return twoGamma;
}
