#include "SimG4Core/GFlash/interface/GflashHadronShowerProfile.h"
#include "SimG4Core/GFlash/interface/GflashTrajectoryPoint.h"

#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/HcalHitMaker.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "FastSimulation/Event/interface/FSimTrack.h"

#include "CLHEP/GenericFunctions/IncompleteGamma.hh"
#include "CLHEP/GenericFunctions/LogGamma.hh"
#include "Randomize.hh"
#include "G4Poisson.hh"
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
  theExportToFastSim = parSet.getParameter<bool>("GflashExportToFastSim");
  theGflashHcalOuter = parSet.getParameter<bool>("GflashHcalOuter");

  theGflashStep = new G4Step();
  theShowino = new GflashShowino();
  theGflashTouchableHandle = new G4TouchableHistory();

  theGflashNavigator = new G4Navigator();

  theHisto = GflashHistogram::instance();
}

GflashHadronShowerProfile::~GflashHadronShowerProfile() 
{
  if(theGflashStep) delete theGflashStep;
  if(theShowino) delete theShowino;
}

void GflashHadronShowerProfile::initialize(const G4FastTrack& fastTrack) {

  //initialize GflashShowino for this track

  theShowino->initialize(fastTrack,theBField);

  //set track and process defined step for the Gflash Step which are common
  //for all hits (GflashStep) assinged at the shower starging point
  theGflashStep->SetTrack(const_cast<G4Track*>(fastTrack.GetPrimaryTrack()));

  theGflashStep->GetPostStepPoint()->SetProcessDefinedStep(const_cast<G4VProcess*>
    (fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()));
  theGflashNavigator->SetWorldVolume(G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume());

}

void GflashHadronShowerProfile::hadronicParameterization()
{
  // The skeleton of this method is based on the fortran code gfshow.F originally written  
  // by S. Peters and G. Grindhammer (also see NIM A290 (1990) 469-488), but longitudinal
  // parameterizations of hadron showers are significantly modified for the CMS calorimeter  

  // unit convention: energy in [GeV] and length in [cm]
  // intrinsic properties of hadronic showers (lateral shower profile)
  
  // The step size of showino along the helix trajectory in cm unit
  G4double showerDepthR50 = 0.0;
  bool firstHcalHit = true;

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

    if(whichCalor == Gflash::kNULL || heightProfile <= 0. ) continue;

    deltaEnergy =  heightProfile*Gflash::divisionStep*energyScale[whichCalor];    
    theShowino->addEnergyDeposited(deltaEnergy);

    //apply sampling fluctuation if showino is inside the sampling calorimeter
    double hadronicFraction = 1.0;
    G4double fluctuatedEnergy = deltaEnergy;

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

   //Gflash is exported to FastSim
    if(theExportToFastSim) {
      double currentDepth = 0.0;
      int ecal = 0;
      if(whichCalor==Gflash::kHB || whichCalor==Gflash::kHE) {
        //calculate the half point of each step of the shower from the surface of the Ecal
        currentDepth = theShowino->getDepthAtShower() + (showerDepth - 0.5*deltaStep);
        bool setHDdepth = theHcalHitMaker->setDepth(currentDepth,true);
        if(!setHDdepth) continue;
	//@@@tune the scale factor for Hcal hits
        theHcalHitMaker->setSpotEnergy(sampleSpotEnergy*1.4);
      }
      else {
        ecal = 1;
        currentDepth = theShowino->getDepthAtShower();
        bool status = theEcalHitMaker->getPads(currentDepth,true);
        if(!status) continue;
	//@@@tune the scale factor for Ecal hits
        theEcalHitMaker->setSpotEnergy(sampleSpotEnergy*1.2);
      }

      for (G4int ispot = 0 ;  ispot < nSpotsInStep ; ispot++) {
        // Compute global position of generated spots with taking into account magnetic field
        // Divide deltaStep into nSpotsInStep and give a spot a global position
        G4double incrementPath = theShowino->getPathLength()
          + (deltaStep/nSpotsInStep)*(ispot+0.5 - 0.5*nSpotsInStep);

        // trajectoryPoint give a spot an imaginary point along the shower development
        GflashTrajectoryPoint trajectoryPoint;
        theShowino->getHelix()->getGflashTrajectoryPoint(trajectoryPoint,incrementPath);

        // Smearing in r according to f(r)= 2.*r*R50**2/(r**2+R50**2)**2
        G4double lateralArm = R50;

        G4double rnunif = G4UniformRand();
        G4double rxPDF = std::sqrt(rnunif/(1.-rnunif));
        G4double rShower = lateralArm*rxPDF;

        //rShower within maxLateralArmforR50
        rShower = std::min(Gflash::maxLateralArmforR50,rShower);
        G4double azimuthalAngle = twopi*G4UniformRand();

        bool result;
        if(whichCalor==Gflash::kHB || whichCalor==Gflash::kHE) {
          result = theHcalHitMaker->addHit(rShower/Gflash::intLength[Gflash::kHB],azimuthalAngle,0);
        }
        else {
          result = theEcalHitMaker->addHit(rShower/Gflash::intLength[Gflash::kESPM],azimuthalAngle,0);
        }
      }
      //end of the branch for FastSim
    } 
    else {
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
	
	aSensitive->Hit(theGflashStep);
	
      } // end of for spot iteration
    }
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
	
	G4int nSpotsInStep = std::max(50,static_cast<int>((160.+40*G4RandGauss::shoot())*std::log(theShowino->getEnergy())+50.));
	
      	double hoFraction = 1.00;
	G4double poissonProb = G4Poisson(1.0);
	
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
	  
	  if( aSensitive == 0 || (std::fabs(spotPosition.getZ()/cm) > Gflash::Zmax[Gflash::kHO]) ) continue;
	  
	  G4String nameCalor = aCurrentVolume->GetName();
	  nameCalor.assign(nameCalor,0,2);
	  if(nameCalor == "HT") spotEnergy *= Gflash::scaleSensitive;
	  
	  theGflashStep->SetTotalEnergyDeposit(spotEnergy);

	  aSensitive->Hit(theGflashStep);
	  
	} // end of for HO spot iteration
      } // end of while for HO longitudinal integration
    }
  }

  //  delete theGflashNavigator;

}

void GflashHadronShowerProfile::loadParameters()
{
  std::cout << "GflashHadronShowerProfile::loadParameters should be implimented for "
            << "each particle type " << std::endl;
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
    
    if(R50>0) {
      double sigmaSq  = std::log(varinanceR50/(R50*R50)+1.0);
      double sigmaR50 = std::sqrt(sigmaSq);
      double meanR50  = std::log(R50) - (sigmaSq/2.);
      
      lateralArm = std::exp(meanR50 + sigmaR50*G4RandGauss::shoot());
    }
  }
  return lateralArm;
}

G4ThreeVector GflashHadronShowerProfile::locateSpotPosition(GflashTrajectoryPoint& point, G4double lateralArm) 
{
  // Smearing in r according to f(r)= 2.*r*R50**2/(r**2+R50**2)**2
  G4double rnunif = G4UniformRand();
  G4double rxPDF = std::sqrt(rnunif/(1.-rnunif));
  G4double rShower = lateralArm*rxPDF;
  
  //rShower within maxLateralArmforR50
  rShower = std::min(Gflash::maxLateralArmforR50,rShower);

  // Uniform smearing in phi
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

G4double GflashHadronShowerProfile::longitudinalProfile(G4double showerDepth, G4double pathLength) {

  G4double heightProfile = 0;
  G4double transDepth = theShowino->getStepLengthToHcal();
  G4ThreeVector pos = theShowino->getPosition();
  G4int showerType = theShowino->getShowerType();
  //  G4double einc = theShowino->getEnergy();

  // Energy in a delta step (dz) = (energy to deposite)*[Gamma(z+dz)-Gamma(z)]*dz
  // where the incomplete Gamma function gives an intergrate probability of the longitudinal 
  // shower up to the shower depth (z).
  // Instead, we use approximated energy; energy in dz = (energy to deposite)*gamma(z)*dz
  // where gamma is the Gamma-distributed probability function

  GflashTrajectoryPoint tempPoint;
  theShowino->getHelix()->getGflashTrajectoryPoint(tempPoint,pathLength);

  if(showerType == 0 || showerType == 1 ) {
    //@@@@change 152 to the 129+22*sin(theta) type of boundary
    if(tempPoint.getPosition().getRho() < 152.0 ) { 
      heightProfile = twoGammaProfile(longEcal,showerDepth,Gflash::kESPM);
    }
    else if (tempPoint.getPosition().getRho() > Gflash::Rmin[Gflash::kHB] ){
      heightProfile = twoGammaProfile(longHcal,showerDepth-transDepth,Gflash::kHB);
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

G4double GflashHadronShowerProfile::hoProfile(G4double pathLength, G4double refDepth) {

  G4double heightProfile = 0;

  GflashTrajectoryPoint tempPoint;
  theShowino->getHelix()->getGflashTrajectoryPoint(tempPoint,pathLength);

  G4double dint = 1.4*Gflash::intLength[Gflash::kHO]*std::sin(tempPoint.getPosition().getTheta());
  heightProfile = std::exp(-1.0*refDepth/dint);

  return heightProfile;
}

void GflashHadronShowerProfile::getFluctuationVector(G4double *lowTriangle, G4double *correlationVector) {

    const G4int dim = Gflash::NPar;

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
  G4int showerType = theShowino->getShowerType();

  G4int numberOfSpots = 0;
  G4double nmean  = 0.0;
  G4double nsigma = 0.0;

  if(showerType == 0 || showerType == 1 || showerType == 4 || showerType == 5 ) {
    if(kCalor == Gflash::kESPM) {
      nmean = 10000 + 5000*log(einc);
      nsigma = 1000;
    }
    if(kCalor == Gflash::kHB) {
      nmean =  5000 + 2500*log(einc);
      nsigma =  500;
    }
  }
  else if (showerType == 2 || showerType == 3 || showerType == 6 || showerType == 7 ) {
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

  numberOfSpots = std::max(500,static_cast<int> (nmean+nsigma*G4RandGauss::shoot()));

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

void GflashHadronShowerProfile::initFastSimCaloHit(EcalHitMaker *aEcalHitMaker, HcalHitMaker *aHcalHitMaker) {
  theEcalHitMaker = aEcalHitMaker;
  theHcalHitMaker = aHcalHitMaker;
}
