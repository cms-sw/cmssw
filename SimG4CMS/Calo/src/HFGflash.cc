#include "SimG4CMS/Calo/interface/HFGflash.h"

#include "G4VPhysicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Navigator.hh"
#include "G4NavigationHistory.hh"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

#include "Randomize.hh"
#include "G4TransportationManager.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4VSensitiveDetector.hh"
#include "G4EventManager.hh"
#include "G4SteppingManager.hh"
#include "G4FastTrack.hh"
#include "G4ParticleTable.hh"

#include "CLHEP/GenericFunctions/IncompleteGamma.hh"

#include "SimG4Core/Application/interface/SteppingAction.h"
#include "SimGeneral/GFlash/interface/GflashEMShowerProfile.h"
#include "SimGeneral/GFlash/interface/GflashTrajectoryPoint.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <math.h>

//#define DebugLog

HFGflash::HFGflash(edm::ParameterSet const & p) {

  edm::ParameterSet m_HF  = p.getParameter<edm::ParameterSet>("HFGflash");
  theBField               = m_HF.getUntrackedParameter<double>("BField", 3.8);
  theWatcherOn            = m_HF.getUntrackedParameter<bool>("WatcherOn",true);
  theFillHisto            = m_HF.getUntrackedParameter<bool>("FillHisto",true);
  edm::LogInfo("HFShower") << "HFGFlash:: Set B-Field to " << theBField
			   << ", WatcherOn to " << theWatcherOn
			   << " and FillHisto to " << theFillHisto;

  theHelix = new GflashTrajectory;
  theGflashStep = new G4Step();
  theGflashNavigator = new G4Navigator();
  //  theGflashNavigator = 0;
  theGflashTouchableHandle = new G4TouchableHistory();

#ifdef DebugLog
  if (theFillHisto) {
    edm::Service<TFileService> tfile;
    if ( tfile.isAvailable() ) {
      TFileDirectory showerDir = tfile->mkdir("GflashEMShowerProfile");

      em_incE = showerDir.make<TH1F>("em_incE","Incoming energy (GeV)",500,0,500.);
      em_ssp_rho    = showerDir.make<TH1F>("em_ssp_rho","Shower starting position;#rho (cm);Number of Events",100,100.0,200.0);
      em_ssp_z      = showerDir.make<TH1F>("em_ssp_z","Shower starting position;z (cm);Number of Events",2000,0.0,2000.0);
      em_long       = showerDir.make<TH1F>("em_long","Longitudinal Profile;Radiation Length;Number of Spots",800,800.0,1600.0);
      em_lateral    = showerDir.make<TH1F>("em_lateral","Lateral Profile;Radiation Length;Moliere Radius",100,0.0,5.0);
      em_2d         = showerDir.make<TH2F>("em_2d","Lateral Profile vs. Shower Depth;Radiation Length;Moliere Radius",800,800.0,1600.0,100,0.0,5.0);
      em_long_sd    = showerDir.make<TH1F>("em_long_sd","Longitudinal Profile in Sensitive Detector;Radiation Length;Number of Spots",800,800.0,1600.0);
      em_lateral_sd = showerDir.make<TH1F>("em_lateral_sd","Lateral Profile vs. Shower Depth in Sensitive Detector;Radiation Length;Moliere Radius",100,0.0,5.0);
      em_2d_sd      = showerDir.make<TH2F>("em_2d_sd","Lateral Profile vs. Shower Depth in Sensitive Detector;Radiation Length;Moliere Radius",800,800.0,1600.0,100,0.0,5.0);
      em_ze         = showerDir.make<TH2F>("em_ze","Profile vs. Energy;Radiation Length;Moliere Radius",800,800.0,1600.0,1000,0.0,1.0);
      em_ratio      = showerDir.make<TH2F>("em_ratio","Profile vs. Energy;Radiation Length;Moliere Radius",800,800.0,1600.0,1000,0.0,100.0);
      em_ratio_selected      = showerDir.make<TH2F>("em_ratio_selected","Profile vs. Energy;Radiation Length;Moliere Radius",800,800.0,1600.0,1000,0.0,100.0);
      em_nSpots_sd  = showerDir.make<TH1F>("em_nSpots_sd","Number of Gflash Spots in Sensitive Detector;Number of Spots;Number of Events",100,0.0,100);
      em_ze_ratio   = showerDir.make<TH1F>("em_ze_ratio","Ratio of Energy and Z Position",1000,0.0,0.001);
    } else {
      theFillHisto = false;
      edm::LogInfo("HFShower") << "HFGFlash::No file is available for saving"
			       << " histos so the flag is set to false";
    }
  }
#endif
  jCalorimeter = Gflash::kHF;

}

HFGflash::~HFGflash() {
  if (theHelix)      delete theHelix;
  if (theGflashStep) delete theGflashStep;
  if (theGflashNavigator) delete theGflashNavigator;
}


std::vector<HFGflash::Hit> HFGflash::gfParameterization(G4Step * aStep,bool & ok,bool onlyLong) {
  double tempZCalo = 26;  // Gflash::Z[jCalorimeter]
  double hfcriticalEnergy = 0.021;  // Gflash::criticalEnergy

  std::vector<HFGflash::Hit> hit;
  HFGflash::Hit oneHit;

  G4StepPoint * preStepPoint  = aStep->GetPreStepPoint(); 
  //G4StepPoint * postStepPoint = aStep->GetPostStepPoint(); 
  G4Track *     track    = aStep->GetTrack();
  // Get Z-direction 
  const G4DynamicParticle *aParticle = track->GetDynamicParticle();
  G4ThreeVector momDir = aParticle->GetMomentumDirection();

  G4ThreeVector hitPoint = preStepPoint->GetPosition();   
  G4String      partType = track->GetDefinition()->GetParticleName();
  //  int           parCode  = track->GetDefinition()->GetPDGEncoding();

  // This part of code is copied from the original GFlash Fortran code.
  // reference : hep-ex/0001020v1

  const G4double energyCutoff     = 1; 
  const G4int    maxNumberOfSpots = 10000000;

  G4ThreeVector showerStartingPosition = track->GetPosition()/cm;
  G4ThreeVector showerMomentum = preStepPoint->GetMomentum()/GeV;
  jCalorimeter = Gflash::kHF;

  G4double logEinc = std::log((preStepPoint->GetTotalEnergy())/GeV);

  G4double y = ((preStepPoint->GetTotalEnergy())/GeV) / hfcriticalEnergy; // y = E/Ec, criticalEnergy is in GeV
  G4double logY = std::log(y);


  G4double nSpots = 93.0 * std::log(tempZCalo) * ((preStepPoint->GetTotalEnergy())/GeV); // total number of spot due linearization
  if(preStepPoint->GetTotalEnergy()/GeV < 1.6)  nSpots = 140.4 * std::log(tempZCalo) * std::pow(((preStepPoint->GetTotalEnergy())/GeV),0.876); 


  //   // implementing magnetic field effects
  double charge = track->GetStep()->GetPreStepPoint()->GetCharge();
  theHelix->initializeTrajectory(showerMomentum,showerStartingPosition,charge,theBField);

  G4double pathLength0 = theHelix->getPathLengthAtZ(showerStartingPosition.getZ());
  G4double pathLength = pathLength0; // this will grow along the shower development

  //--- 2.2  Fix intrinsic properties of em. showers.

  G4double fluctuatedTmax = std::log(logY - 0.7157);
  G4double fluctuatedAlpha= std::log(0.7996 +(0.4581 + 1.8628/tempZCalo)*logY);

  G4double sigmaTmax = 1.0/( -1.4  + 1.26 * logY);
  G4double sigmaAlpha = 1.0/( -0.58 + 0.86 * logY);
  G4double rho = 0.705  - 0.023 * logY;
  G4double sqrtPL = std::sqrt((1.0+rho)/2.0);
  G4double sqrtLE = std::sqrt((1.0-rho)/2.0);

  G4double norm1 = G4RandGauss::shoot();
  G4double norm2 = G4RandGauss::shoot();
  G4double tempTmax = fluctuatedTmax + sigmaTmax*(sqrtPL*norm1 + sqrtLE*norm2);
  G4double tempAlpha = fluctuatedAlpha + sigmaAlpha*(sqrtPL*norm1 - sqrtLE*norm2);

  // tmax, alpha, beta : parameters of gamma distribution
  G4double tmax = std::exp(tempTmax);
  G4double alpha = std::exp(tempAlpha);
  G4double beta = (alpha - 1.0)/tmax;

  if (!alpha)          return hit; 
  if (!beta)           return hit;
  if (alpha < 0.00001) return hit;
  if (beta < 0.00001)  return hit;
 
  // spot fluctuations are added to tmax, alpha, beta
  G4double averageTmax = logY-0.858;
  G4double averageAlpha = 0.21+(0.492+2.38/tempZCalo)*logY;
  G4double spotTmax  = averageTmax * (0.698 + .00212*tempZCalo);
  G4double spotAlpha= averageAlpha * (0.639 + .00334*tempZCalo);
  G4double spotBeta = (spotAlpha-1.0)/spotTmax;

  if (!spotAlpha)	   return hit;
  if (!spotBeta)	   return hit;
  if (spotAlpha < 0.00001) return hit;
  if (spotBeta < 0.00001)  return hit;

#ifdef DebugLog  
  LogDebug("HFShower") << "Incoming energy = " << ((preStepPoint->GetTotalEnergy())/GeV) << " Position (rho,z) = (" << showerStartingPosition.rho() << ", " << showerStartingPosition.z() << ")";

  if(theFillHisto) {
    em_incE->Fill(((preStepPoint->GetTotalEnergy())/GeV));
    em_ssp_rho->Fill(showerStartingPosition.rho());
    em_ssp_z->Fill(std::abs(showerStartingPosition.z()));
  }
#endif
  //  parameters for lateral distribution and fluctuation
  G4double z1=0.0251+0.00319*logEinc;
  G4double z2=0.1162-0.000381*tempZCalo;

  G4double k1=0.659 - 0.00309 * tempZCalo;
  G4double k2=0.645;
  G4double k3=-2.59;
  G4double k4=0.3585+ 0.0421*logEinc;

  G4double p1=2.623 -0.00094*tempZCalo;
  G4double p2=0.401 +0.00187*tempZCalo;
  G4double p3=1.313 -0.0686*logEinc;

  //   // @@@ dwjang, intial tuning by comparing 20-150GeV TB data
  //   // the width of energy response is not yet tuned.
  G4double e25Scale = 1.03551;
  z1 *= 9.76972e-01 - 3.85026e-01 * std::tanh(1.82790e+00*std::log(((preStepPoint->GetTotalEnergy())/GeV)) - 3.66237e+00);
  p1 *= 0.96;

  G4double stepLengthLeft = 10000;
  G4int    nSpots_sd = 0; // count total number of spots in SD
  G4double zInX0 = 0.0; // shower depth in X0 unit
  G4double deltaZInX0 = 0.0; // segment of depth in X0 unit
  G4double deltaZ = 0.0; // segment of depth in cm
  G4double stepLengthLeftInX0 = 0.0; // step length left in X0 unit

  const G4double divisionStepInX0 = 0.1; //step size in X0 unit
  G4double energy = ((preStepPoint->GetTotalEnergy())/GeV); // energy left in GeV

  Genfun::IncompleteGamma gammaDist;

  G4double energyInGamma = 0.0; // energy in a specific depth(z) according to Gamma distribution
  G4double preEnergyInGamma = 0.0; // energy calculated in a previous depth
  G4double sigmaInGamma  = 0.; // sigma of energy in a specific depth(z) according to Gamma distribution
  G4double preSigmaInGamma = 0.0; // sigma of energy in a previous depth

  //energy segment in Gamma distribution of shower in each step  
  G4double deltaEnergy =0.0 ; // energy in deltaZ
  G4int spotCounter = 0; // keep track of number of spots generated

  //step increment along the shower direction
  G4double deltaStep = 0.0;

  // Uniqueness of G4Step is important otherwise hits won't be created.
  G4double timeGlobal = track->GetStep()->GetPreStepPoint()->GetGlobalTime();

  // this needs to be deleted manually at the end of this loop.
  //  theGflashNavigator = new G4Navigator();
  theGflashNavigator->SetWorldVolume(G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume());

  //   // loop for longitudinal integration

#ifdef DebugLog  
  LogDebug("HFShower") << " Energy = " << energy << " Step Length Left = "  << stepLengthLeft;
#endif
  while(energy > 0.0 && stepLengthLeft > 0.0) { 
    stepLengthLeftInX0 = stepLengthLeft / Gflash::radLength[jCalorimeter];

    if ( stepLengthLeftInX0 < divisionStepInX0 ) {
      deltaZInX0 = stepLengthLeftInX0;
      deltaZ     = deltaZInX0 * Gflash::radLength[jCalorimeter];
      stepLengthLeft = 0.0;
    } else {
      deltaZInX0 = divisionStepInX0;
      deltaZ     = deltaZInX0 * Gflash::radLength[jCalorimeter];
      stepLengthLeft -= deltaZ;
    }

    zInX0 += deltaZInX0;
    
#ifdef DebugLog  
    LogDebug("HFShower") << " zInX0 = " << zInX0 << " spotBeta*zInX0 = " << spotBeta*zInX0;
#endif
    if ((!zInX0) || (!spotBeta*zInX0) || (zInX0 < 0.01) || 
	(spotBeta*zInX0 < 0.00001) || (!zInX0*beta) || (zInX0*beta < 0.00001)) 
      return hit;

    G4int nSpotsInStep = 0;

#ifdef DebugLog  
    LogDebug("HFShower") << " Energy - Energy Cut off = " << energy - energyCutoff;
#endif

    if ( energy > energyCutoff  ) {
      preEnergyInGamma  = energyInGamma;
      gammaDist.a().setValue(alpha);  //alpha
      
      energyInGamma = gammaDist(beta*zInX0); //beta
      G4double energyInDeltaZ  = energyInGamma - preEnergyInGamma;
      deltaEnergy   = std::min(energy,((preStepPoint->GetTotalEnergy())/GeV)*energyInDeltaZ);
 
      preSigmaInGamma  = sigmaInGamma;
      gammaDist.a().setValue(spotAlpha);  //alpha spot
      sigmaInGamma = gammaDist(spotBeta*zInX0); //beta spot
      nSpotsInStep = std::max(1,int(nSpots * (sigmaInGamma - preSigmaInGamma)));
    } else {
      deltaEnergy = energy;
      preSigmaInGamma  = sigmaInGamma;
      nSpotsInStep = std::max(1,int(nSpots * (1.0 - preSigmaInGamma)));
    }

    if ( deltaEnergy > energy || (energy-deltaEnergy) < energyCutoff ) deltaEnergy = energy;

    energy  -= deltaEnergy;

    if ( spotCounter+nSpotsInStep > maxNumberOfSpots ) {
      nSpotsInStep = maxNumberOfSpots - spotCounter;
      if (nSpotsInStep < 1) nSpotsInStep = 1;
    }


    //     // It begins with 0.5 of deltaZ and then icreases by 1 deltaZ
    deltaStep  += 0.5*deltaZ;
    pathLength += deltaStep;
    deltaStep   =  0.5*deltaZ;


    //lateral shape and fluctuations for  homogenous calo.
    G4double tScale = tmax *alpha/(alpha-1.0) * (std::exp(fluctuatedAlpha)-1.0)/std::exp(fluctuatedAlpha);
    G4double tau = std::min(10.0,(zInX0 - 0.5*deltaZInX0)/tScale);
    G4double rCore = z1 + z2 * tau; 
    G4double rTail = k1 *( std::exp(k3*(tau-k2)) + std::exp(k4*(tau-k2))); // @@ check RT3 sign
    G4double p23 = (p2 - tau)/p3;
    G4double probabilityWeight = p1 *  std::exp( p23 - std::exp(p23) );


    // Deposition of spots according to lateral distr.
    // Apply absolute energy scale
    // Convert into MeV unit
    G4double emSpotEnergy = deltaEnergy / nSpotsInStep * e25Scale * GeV;


#ifdef DebugLog  
    LogDebug("HFShower") << " nSpotsInStep = " << nSpotsInStep;
#endif



    for (G4int ispot = 0 ;  ispot < nSpotsInStep ; ispot++) {
      spotCounter++;
      G4double u1 = G4UniformRand();
      G4double u2 = G4UniformRand();
      G4double rInRM = 0.0;
      
      if (u1 < probabilityWeight) {
	rInRM = rCore * std::sqrt( u2/(1.0-u2) );
      } else {
	rInRM = rTail * std::sqrt( u2/(1.0-u2) );
      }
  
      G4double rShower =  rInRM * Gflash::rMoliere[jCalorimeter];
      
      //Uniform & random rotation of spot along the azimuthal angle
      G4double azimuthalAngle = twopi*G4UniformRand();
      
      //Compute global position of generated spots with taking into account magnetic field
      //Divide deltaZ into nSpotsInStep and give a spot a global position
      G4double incrementPath = (deltaZ/nSpotsInStep)*(ispot+0.5 - 0.5*nSpotsInStep);

      // trajectoryPoint give a spot an imaginary point along the shower development
      GflashTrajectoryPoint trajectoryPoint;
      theHelix->getGflashTrajectoryPoint(trajectoryPoint,pathLength+incrementPath);


      G4ThreeVector SpotPosition0 = trajectoryPoint.getPosition() + rShower*std::cos(azimuthalAngle)*trajectoryPoint.getOrthogonalUnitVector() + rShower*std::sin(azimuthalAngle)*trajectoryPoint.getCrossUnitVector();

      //!V.Ivanchenko - not clear if it is correct
      // Convert into mm unit
      SpotPosition0 *= cm;


      //---------------------------------------------------
      // fill a fake step to send it to hit maker
      //---------------------------------------------------

      // to make a different time for each fake step. (0.03 nsec is corresponding to 1cm step size)
      timeGlobal += 0.0001*nanosecond;

      //fill equivalent changes to a (fake) step associated with a spot 
      
      G4double zInX0_spot = std::abs(pathLength+incrementPath - pathLength0)/Gflash::radLength[jCalorimeter];

#ifdef DebugLog  
      LogDebug("HFShower") <<  "zInX0_spot,emSpotEnergy/GeV =" << zInX0_spot << " , " << emSpotEnergy/GeV <<  "emSpotEnergy/GeV =" << emSpotEnergy/GeV;
#endif

      if ((!zInX0_spot) || (zInX0_spot < 0)) continue;
      if ((!emSpotEnergy/GeV) ||  (emSpotEnergy < 0)) continue;
      if ((!rShower/Gflash::rMoliere[jCalorimeter]) || (rShower/Gflash::rMoliere[jCalorimeter] < 0)) continue;

      oneHit.depth    = 1;

#ifdef DebugLog
      if (theFillHisto) {
	em_long->Fill(SpotPosition0.z()/cm,emSpotEnergy/GeV);
	em_lateral->Fill(rShower/Gflash::rMoliere[jCalorimeter],emSpotEnergy/GeV);
	em_2d->Fill(SpotPosition0.z()/cm,rShower/Gflash::rMoliere[jCalorimeter],emSpotEnergy/GeV);
      }
#endif

      //!V.Ivanchenko what this means??
      //if(SpotPosition0 == 0) continue;

      double energyratio = emSpotEnergy/(preStepPoint->GetTotalEnergy()/(nSpots*e25Scale));

      if (emSpotEnergy/GeV < 0.0001) continue;
      if (energyratio > 80) continue;

      double zshift =0;
      if(SpotPosition0.z() > 0) zshift = 18;
      if(SpotPosition0.z() < 0) zshift = -18;


      G4ThreeVector gfshift(0,0,zshift*(pow(100,0.1)/pow(preStepPoint->GetTotalEnergy()/GeV,0.1)));

      G4ThreeVector SpotPosition = gfshift + SpotPosition0;

      double LengthWeight = std::fabs(std::pow(SpotPosition0.z()/11370,1));
      if (G4UniformRand()>  0.0021 * energyratio * LengthWeight) continue;

      oneHit.position = SpotPosition;
      oneHit.time     = timeGlobal;
      oneHit.edep     = emSpotEnergy/GeV;
      hit.push_back(oneHit);
      nSpots_sd++;
      
    } // end of for spot iteration

  } // end of while for longitudinal integration
#ifdef DebugLog
  if (theFillHisto) {
    em_nSpots_sd->Fill(nSpots_sd);
  }
#endif
  //  delete theGflashNavigator;
  return hit;
}
