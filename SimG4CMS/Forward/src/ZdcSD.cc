///////////////////////////////////////////////////////////////////////////////
// File: ZdcSD.cc
// Date: 03.01
// Description: Sensitive Detector class for Zdc
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Forward/interface/ZdcSD.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#include "G4ios.hh"
#include "G4Poisson.hh"
#include "G4Cerenkov.hh"

#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Random/Randomize.h"

ZdcSD::ZdcSD(G4String name, const DDCompactView & cpv,
	     edm::ParameterSet const & p,const SimTrackManager* manager) : 
  CaloSD(name, cpv, p, manager), numberingScheme(0) {
  edm::ParameterSet m_ZdcSD = p.getParameter<edm::ParameterSet>("ZdcSD");
  verbosity  = m_ZdcSD.getParameter<int>("Verbosity");
  int verbn  = verbosity/10;
  verbosity %= 10;
  setNumberingScheme(new ZdcNumberingScheme(verbn));

  edm::LogInfo("ForwardSim")
    << "***************************************************\n"
    << "*                                                 *\n"
    << "* Constructing a ZdcSD  with name " << name << "\n"
    << "*                                                 *\n"
    << "***************************************************";
}

ZdcSD::~ZdcSD() {}

double ZdcSD::getEnergyDeposit(G4Step * aStep, edm::ParameterSet const & p ) {

  float NCherPhot = 0.;

  if (aStep == NULL) {
    LogDebug("ForwardSim") << "ZdcSD::  getEnergyDeposit: aStep is NULL!";
    return 0;
  }
  else {
    // preStepPoint information
    G4SteppingControl  stepControlFlag = aStep->GetControlFlag();
    G4StepPoint*       preStepPoint = aStep->GetPreStepPoint();
    G4VPhysicalVolume* currentPV    = preStepPoint->GetPhysicalVolume();
    G4String           nameVolume   = currentPV->GetName();

    G4ThreeVector      hitPoint = preStepPoint->GetPosition();	
    G4ThreeVector      hit_mom = preStepPoint->GetMomentumDirection();
    G4double           stepL = aStep->GetStepLength()/cm;
    G4double           beta     = preStepPoint->GetBeta();
    G4double           charge   = preStepPoint->GetCharge();

    // G4VProcess*        curprocess   = preStepPoint->GetProcessDefinedStep();
    // G4String           namePr   = preStepPoint->GetProcessDefinedStep()->GetProcessName();
    // G4LogicalVolume*   lv    = currentPV->GetLogicalVolume();
    // G4Material*        mat   = lv->GetMaterial();
    // G4double           rad   = mat->GetRadlen();

    // postStepPoint information
    G4StepPoint* postStepPoint = aStep->GetPostStepPoint();   
    G4VPhysicalVolume* postPV = postStepPoint->GetPhysicalVolume();
    G4String postnameVolume = postPV->GetName();

    // theTrack information
    G4Track* theTrack = aStep->GetTrack();   
    G4String particleType = theTrack->GetDefinition()->GetParticleName();
    G4int primaryID = theTrack->GetTrackID();
    G4double entot = theTrack->GetTotalEnergy();
    G4ThreeVector vert_mom = theTrack->GetVertexMomentumDirection();
    G4ThreeVector localPoint = theTrack->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPoint);

    // calculations
    float costheta = vert_mom.z()/sqrt(vert_mom.x()*vert_mom.x()+
				       vert_mom.y()*vert_mom.y()+
				       vert_mom.z()*vert_mom.z());
    float theta = acos(min(max(costheta,float(-1.)),float(1.)));
    float eta = -log(tan(theta/2));
    float phi = -100.;
    if (vert_mom.x() != 0) phi = atan2(vert_mom.y(),vert_mom.x()); 
    if (phi < 0.) phi += twopi;

    // Get the total energy deposit
    double stepE   = aStep->GetTotalEnergyDeposit();
    LogDebug("ForwardSim") 
      << "ZdcSD::  getEnergyDeposit: "
      <<"*****************HHHHHHHHHHHHHHHHHHHHHHHHHHLLLLLLLLLlllllllllll&&&&&&&&&&\n"
      << "  preStepPoint: " << nameVolume << "," << stepL << "," << stepE 
      << "," << beta << "," << charge << "\n"
      << "  postStepPoint: " << postnameVolume << "," << costheta << "," 
      << theta << "," << eta << "," << phi << "," << particleType << "," 
      << primaryID;

    float bThreshold = 0.67;
    int status = 0;
    if ((beta > bThreshold) && (charge != 0) && (nameVolume == "ZDC_EMFiber" || nameVolume == "ZDC_HadFiber")) {
      status = 1;
      LogDebug("ForwardSim") << "ZdcSD::  getEnergyDeposit:  pass "; 

      float nMedium = 1.4925;
      // float photEnSpectrDL = 10714.285714;
      //       photEnSpectrDL = (1./400.nm-1./700.nm)*10000000.cm/nm; /* cm-1  */

      float photEnSpectrDE = 1.24;
      // E = 2pi*(1./137.)*(eV*cm/370.)/lambda = 12.389184*(eV*cm)/lambda
      // Emax = 12.389184*(eV*cm)/400nm*10-7cm/nm  = 3.01 eV
      // Emin = 12.389184*(eV*cm)/700nm*10-7cm/nm  = 1.77 eV
      // delE = Emax - Emin = 1.24 eV

      float effPMTandTransport = 0.15;

      // Check these values
      float thFullRefl = 23.;
      float thFullReflRad = thFullRefl*pi/180.;

      edm::ParameterSet m_ZdcSD = p.getParameter<edm::ParameterSet>("ZdcSD");
      thFibDir  = m_ZdcSD.getParameter<double>("FiberDirection");
      //float thFibDir = 90.;
      float thFibDirRad = thFibDir*pi/180.;

      // at which theta the point is located:
      //   float th1 = hitPoint.theta();

      // theta of charged particle in LabRF(hit momentum direction):
      float costh = hit_mom.z()/sqrt(hit_mom.x()*hit_mom.x()+
				     hit_mom.y()*hit_mom.y()+
				     hit_mom.z()*hit_mom.z());
      float th = acos(min(max(costh,float(-1.)),float(1.)));
      // just in case (can do both standard ranges of phi):
      if (th < 0.) th += twopi;

      // theta of cone with Cherenkov photons w.r.t.direction of charged part.:
      float costhcher =1./(nMedium*beta);
      float thcher = acos(min(max(costhcher,float(-1.)),float(1.)));

      // diff thetas of charged part. and quartz direction in LabRF:
      float DelFibPart = fabs(th - thFibDirRad);

      // define real distances:
      float d = fabs(tan(th)-tan(thFibDirRad));   

      // float a = fabs(tan(thFibDirRad)-tan(thFibDirRad+thFullReflRad));   
      // float r = fabs(tan(th)-tan(th+thcher));   
      float a = tan(thFibDirRad)+tan(fabs(thFibDirRad-thFullReflRad));   
      float r = tan(th)+tan(fabs(th-thcher));   
      
      // std::cout.testOut << "  d=|tan(" << th << ")-tan(" << thFibDirRad << ")| "
      //	      << "=|" << tan(th) << "-" << tan(thFibDirRad) << "| = " << d;
      // std::cout.testOut << "  a=tan(" << thFibDirRad << ")=" << tan(thFibDirRad) 
      //              << " + tan(|" << thFibDirRad << " - " << thFullReflRad << "|)="
      //              << tan(fabs(thFibDirRad-thFullReflRad)) << " = " << a;
      // std::cout.testOut << "  r=tan(" << th << ")=" << tan(th) << " + tan(|" << th 
      //              << " - " << thcher << "|)=" << tan(fabs(th-thcher)) << " = " << r;

      // define losses d_qz in cone of full reflection inside quartz direction
      float d_qz = -1;
      float variant = -1;

      // if (d > (r+a))
      if (DelFibPart > (thFullReflRad + thcher) ) {
        variant = 0.; d_qz = 0.;
      }
      else {
        // if ((DelFibPart + thcher) < thFullReflRad )  [(d+r) < a]
	if ((th + thcher) < (thFibDirRad+thFullReflRad) && (th - thcher) > (thFibDirRad-thFullReflRad) ) {
	  variant = 1.; d_qz = 1.;
	}
	else {
          // if ((thcher - DelFibPart ) > thFullReflRad )  [(r-d) > a]
	  if ((thFibDirRad + thFullReflRad) < (th + thcher) && (thFibDirRad - thFullReflRad) > (th - thcher) ) {
            variant = 2.; d_qz = 0.;
	  }
          else {
            // if ((thcher + DelFibPart ) > thFullReflRad && thcher < (DelFibPart+thFullReflRad) ) {  [(r+d) > a && (r-d) < a)]
            variant = 3.; // d_qz is calculated below

            // use crossed length of circles(cone projection) - dC1/dC2 : 
	    float arg_arcos = 0.;
	    float tan_arcos = 2.*a*d;
	    if (tan_arcos != 0.) arg_arcos =(r*r-a*a-d*d)/tan_arcos; 
            // std::cout.testOut << "  d_qz: " << r << "," << a << "," << d << " " << tan_arcos << " " << arg_arcos;
	    arg_arcos = fabs(arg_arcos);
            // std::cout.testOut << "," << arg_arcos;
	    float th_arcos = acos(min(max(arg_arcos,float(-1.)),float(1.)));
            // std::cout.testOut << " " << th_arcos;
	    d_qz = th_arcos/pi/2.;
            // std::cout.testOut << " " << d_qz;
	    d_qz = fabs(d_qz);
            // std::cout.testOut << "," << d_qz;
	  }
	}
      }

      //  std::cout<< std::endl;
      double meanNCherPhot = 0.;
      G4int poissNCherPhot = 0;
      if (d_qz > 0) {
	meanNCherPhot = 370.*charge*charge*( 1. - 1./(nMedium*nMedium*beta*beta) ) * photEnSpectrDE * stepL;

	// dLamdX:  meanNCherPhot = (2.*pi/137.)*charge*charge* 
	//                          ( 1. - 1./(nMedium*nMedium*beta*beta) ) * photEnSpectrDL * stepL;

	// Poisson:  long poissNCherPhot = RandPoisson::shoot(meanNCherPhot);
	poissNCherPhot = (G4int) G4Poisson(meanNCherPhot);

	if (poissNCherPhot < 0) poissNCherPhot = 0; 

	// NCherPhot = meanNCherPhot;
	NCherPhot = poissNCherPhot * effPMTandTransport * d_qz;
      }

      LogDebug("ForwardSim") 
	<< "ZdcSD::  getEnergyDeposit:  gED: "
	<< stepE
	<< "," << costh
	<< "," << th
	<< "," << costhcher
	<< "," << thcher
	<< "," << DelFibPart
	<< "," << d
	<< "," << a
	<< "," << r
	<< "," << hitPoint
	<< "," << hit_mom
	<< "," << stepControlFlag
	<< "," << entot
	<< "," << vert_mom
	<< "," << localPoint
	<< "," << charge
	<< "," << beta
	<< "," << stepL
	<< "," << d_qz
	<< "," << variant
	<< "," << meanNCherPhot
	<< "," << poissNCherPhot
	<< "," << NCherPhot;
      // --constants-----------------
      // << "," << photEnSpectrDE
      // << "," << nMedium
      // << "," << bThreshold
      // << "," << thFibDirRad
      // << "," << thFullReflRad
      // << "," << effPMTandTransport
      // --other variables-----------
      // << "," << curprocess
      // << "," << nameProcess
      // << "," << name
      // << "," << rad
      // << "," << mat

    }
    else {
      // determine failure mode: beta, charge, and/or nameVolume
      status = 0;
      if (beta <= bThreshold)
        LogDebug("ForwardSim") 
	  << "ZdcSD::  getEnergyDeposit: fail beta=" << beta;
      if (charge == 0)
        LogDebug("ForwardSim") 
	  << "ZdcSD::  getEnergyDeposit: fail charge=0";
      if ( !(nameVolume == "ZDC_EMFiber" || nameVolume == "ZDC_HadFiber") )
        LogDebug("ForwardSim") 
	  << "ZdcSD::  getEnergyDeposit: fail nv=" << nameVolume;
    }


    return NCherPhot;
  } 
}

uint32_t ZdcSD::setDetUnitId(G4Step* aStep) {
  uint32_t returnNumber = 0;
  if(numberingScheme != 0)returnNumber = numberingScheme->getUnitID(aStep);
  // edm: return (numberingScheme == 0 ? 0 : numberingScheme->getUnitID(aStep));
  return returnNumber;
}

void ZdcSD::setNumberingScheme(ZdcNumberingScheme* scheme) {
  if (scheme != 0) {
    edm::LogInfo("ForwardSim") << "ZdcSD: updates numbering scheme for " 
			       << GetName();
    if (numberingScheme) delete numberingScheme;
    numberingScheme = scheme;
  }
}



