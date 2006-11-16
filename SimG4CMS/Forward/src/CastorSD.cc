///////////////////////////////////////////////////////////////////////////////
// File: CastorSD.cc
// Date: 02.04
// UpDate: 07.04 - C3TF & C4TF semi-trapezoid added
// Description: Sensitive Detector class for Castor
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Forward/interface/CastorSD.h"
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

#define debug

CastorSD::CastorSD(G4String name, const DDCompactView & cpv,
		   edm::ParameterSet const & p, 
		   const SimTrackManager* manager) : 
  CaloSD(name, cpv, p, manager), numberingScheme(0) {
  
  edm::ParameterSet m_CastorSD = p.getParameter<edm::ParameterSet>("CastorSD");

  setNumberingScheme(new CastorNumberingScheme());
  edm::LogInfo("ForwardSim") 
    << "***************************************************\n"
    << "*                                                 *\n" 
    << "* Constructing a CastorSD  with name " << GetName() << "\n"
    << "*                                                 *\n"
    << "***************************************************";
}

CastorSD::~CastorSD() {}

double CastorSD::getEnergyDeposit(G4Step * aStep) {
  
  float NCherPhot = 0.;

  if (aStep == NULL) {
    return 0;
  } else {
// preStepPoint information *********************************************

    G4SteppingControl  stepControlFlag = aStep->GetControlFlag();
    G4StepPoint*       preStepPoint = aStep->GetPreStepPoint();
    G4VPhysicalVolume* currentPV    = preStepPoint->GetPhysicalVolume();
    G4String           name   = currentPV->GetName();
    std::string        nameVolume;
    nameVolume.assign(name,0,4);



    G4ThreeVector      hitPoint = preStepPoint->GetPosition();	
    G4ThreeVector      hit_mom = preStepPoint->GetMomentumDirection();
    G4double           stepl = aStep->GetStepLength()/cm;
    G4double           beta     = preStepPoint->GetBeta();
    G4double           charge   = preStepPoint->GetCharge();
//    G4VProcess*        curprocess   = preStepPoint->GetProcessDefinedStep();
//    G4String           namePr   = preStepPoint->GetProcessDefinedStep()->GetProcessName();
//     std::string nameProcess;
//     nameProcess.assign(namePr,0,4);

//   G4LogicalVolume*   lv    = currentPV->GetLogicalVolume();
//   G4Material*        mat   = lv->GetMaterial();
//   G4double           rad   = mat->GetRadlen();


// postStepPoint information *********************************************
    G4StepPoint* postStepPoint= aStep->GetPostStepPoint();   
    G4VPhysicalVolume* postPV    = postStepPoint->GetPhysicalVolume();
    G4String           postname   = postPV->GetName();
    std::string        postnameVolume;
    postnameVolume.assign(postname,0,4);


// theTrack information  *************************************************
    G4Track*        theTrack = aStep->GetTrack();   
    G4double        entot    = theTrack->GetTotalEnergy();
    G4ThreeVector   vert_mom = theTrack->GetVertexMomentumDirection();

    G4ThreeVector  localPoint = theTrack->GetTouchable()->GetHistory()->
      GetTopTransform().TransformPoint(hitPoint);


// calculations...       *************************************************

    float costheta =vert_mom.z()/sqrt(vert_mom.x()*vert_mom.x()+
				      vert_mom.y()*vert_mom.y()+
				      vert_mom.z()*vert_mom.z());
    float theta = acos(min(max(costheta,float(-1.)),float(1.)));
    float eta = -log(tan(theta/2));
    float phi = -100.;
    if (vert_mom.x() != 0) phi = atan2(vert_mom.y(),vert_mom.x()); 
    if (phi < 0.) phi += twopi;
    G4String       particleType = theTrack->GetDefinition()->GetParticleName();
    G4int          primaryID    = theTrack->GetTrackID();
// *************************************************


// *************************************************
    double edep   = aStep->GetTotalEnergyDeposit();

// *************************************************


// *************************************************
// take into account light collection curve for plate
//   double weight = curve_Castor(nameVolume, preStepPoint);
//   double edep   = aStep->GetTotalEnergyDeposit() * weight;
// *************************************************


// *************************************************
/*    comments for sensitive volumes:      
C001 ...-... CP06,CBU1 ...-...CALM --- > fibres and bundle 
                                                 for first release of CASTOR
CASF  --- > quartz plate  for first and second releases of CASTOR  
GF2Q, GFNQ, GR2Q, GRNQ 
              for tests with my own test geometry of HF (on ask of Gavrilov)
C3TF, C4TF - for third release of CASTOR
 */
    double meanNCherPhot;

    if(nameVolume == "C001" || nameVolume == "C002" || nameVolume == "C003" ||
       nameVolume == "C004" || nameVolume == "C005" || nameVolume == "C006" ||
       nameVolume == "C007" || nameVolume == "C008" || nameVolume == "C009" ||
       nameVolume == "C010" || nameVolume == "C011" || nameVolume == "C012" || 
       nameVolume == "C013" || nameVolume == "C014" || nameVolume == "CP01" || 
       nameVolume == "CP02" || nameVolume == "CP03" || nameVolume == "CP04" ||
       nameVolume == "CP05" || nameVolume == "CP06" || nameVolume == "CASF" ||
       nameVolume == "CBU1" || nameVolume == "CBU2" || nameVolume == "CBU3" ||
       nameVolume == "CBU4" || nameVolume == "CBU5" || nameVolume == "CBU6" ||
       nameVolume == "CALM" || nameVolume == "GF2Q" || nameVolume == "GFNQ" ||
       nameVolume == "GR2Q" || nameVolume == "GRNQ" ||
       nameVolume == "C3TF" || nameVolume == "C4TF") {

      float bThreshold = 0.67;
      float nMedium = 1.4925;
  //  float photEnSpectrDL = (1./400.nm-1./700.nm)*10000000.cm/nm; /* cm-1  */
  //  float photEnSpectrDL = 10714.285714;

      float photEnSpectrDE = 1.24;    /* see below   */
  /* E = 2pi*(1./137.)*(eV*cm/370.)/lambda  =     */
  /*   = 12.389184*(eV*cm)/lambda                 */
  /* Emax = 12.389184*(eV*cm)/400nm*10-7cm/nm  = 3.01 eV     */
  /* Emin = 12.389184*(eV*cm)/700nm*10-7cm/nm  = 1.77 eV     */
  /* delE = Emax - Emin = 1.24 eV  --> */
/*   */
/* default for Castor nameVolume  == "CASF" or (C3TF & C4TF)  */
      float effPMTandTransport = 0.15;
/* for test HF geometry volumes:   */
      if(nameVolume == "GF2Q" || nameVolume == "GFNQ" || 
	 nameVolume == "GR2Q" || nameVolume == "GRNQ")
	effPMTandTransport = 0.15;

      float thFullRefl = 23.;  /* 23.dergee */
      float thFullReflRad = thFullRefl*pi/180.;

/* default for Castor nameVolume  == "CASF" or (C3TF & C4TF)  */
      float thFibDir = 45.;  /* .dergee */
/* for test HF geometry volumes:   */
      if(nameVolume == "GF2Q" || nameVolume == "GFNQ" ||
	 nameVolume == "GR2Q" || nameVolume == "GRNQ")
	thFibDir = 0.0; /* .dergee */
/*   */
      float thFibDirRad = thFibDir*pi/180.;
/*   */
/*   */

  // at which theta the point is located:
  //  float th1    = hitPoint.theta();

  // theta of charged particle in LabRF(hit momentum direction):
      float costh =hit_mom.z()/sqrt(hit_mom.x()*hit_mom.x()+
				    hit_mom.y()*hit_mom.y()+
				    hit_mom.z()*hit_mom.z());
      float th = acos(min(max(costh,float(-1.)),float(1.)));

    // just in case (can do bot use):
      if (th < 0.) th += twopi;

      float thgrad = th*180./pi;


  // theta of cone with Cherenkov photons w.r.t.direction of charged part.:
      float costhcher =1./(nMedium*beta);
      float thcher = acos(min(max(costhcher,float(-1.)),float(1.)));
      float thchergrad = thcher*180./pi;

  // diff thetas of charged part. and quartz direction in LabRF:
      float DelFibPart = fabs(th - thFibDirRad);
      float DelFibPartgrad = DelFibPart*180./pi;



  // define real distances:
      float d = fabs(tan(th)-tan(thFibDirRad));   

//  float a = fabs(tan(thFibDirRad)-tan(thFibDirRad+thFullReflRad));   
//  float r = fabs(tan(th)-tan(th+thcher));   

      float a = tan(thFibDirRad)+tan(fabs(thFibDirRad-thFullReflRad));   
      float r = tan(th)+tan(fabs(th-thcher));   


  // define losses d_qz in cone of full reflection inside quartz direction
      float d_qz;
      float variant;

      if(DelFibPart > (thFullReflRad + thcher) ) {d_qz = 0.; variant=0.;}
      //	if(d > (r+a) ) {d_qz = 0.; variant=0.;}
      else {
	if((th + thcher) < (thFibDirRad+thFullReflRad) && 
           (th - thcher) > (thFibDirRad-thFullReflRad) 
	   ) {d_qz = 1.; variant=1.;}
	// if((DelFibPart + thcher) < thFullReflRad ) {d_qz = 1.; variant=1.;}
	// if((d+r) < a ) {d_qz = 1.; variant=1.;}
	else {
	  if((thFibDirRad + thFullReflRad) < (th + thcher) && 
	     (thFibDirRad - thFullReflRad) > (th - thcher) ) {
	    // if((thcher - DelFibPart ) > thFullReflRad ) 
	    // if((r-d ) > a ) 
	    d_qz = 0.; variant=2.;

	  } else {
	    // if((thcher + DelFibPart ) > thFullReflRad && 
	    //    thcher < (DelFibPart+thFullReflRad) ) 
	    //      {
	    d_qz = 0.; variant=3.;


	    // use crossed length of circles(cone projection)
	    // dC1/dC2 : 
	    float arg_arcos = 0.;
	    float tan_arcos = 2.*a*d;
	    if(tan_arcos != 0.) arg_arcos =(r*r-a*a-d*d)/tan_arcos; 
	    arg_arcos = fabs(arg_arcos);
	    float th_arcos = acos(min(max(arg_arcos,float(-1.)),float(1.)));
	    d_qz = th_arcos/pi/2.;
	    d_qz = fabs(d_qz);



	    //	    }
	    //             else
	    //  	     {
	    //                d_qz = 0.; variant=4.;
	    //#ifdef debug
	    // std::cout <<" ===============>variant 4 information: <===== " <<std::endl;
	    // std::cout <<" !!!!!!!!!!!!!!!!!!!!!!  variant = " << variant  <<std::endl;
	    //#endif 
	    //
	    // 	     }
	  }
	}
      }


  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      if(charge != 0. && beta > bThreshold && d_qz != 0. ) {

	meanNCherPhot = 370.*charge*charge* 
	  ( 1. - 1./(nMedium*nMedium*beta*beta) )*
	  photEnSpectrDE*stepl;

	// dLamdX:
	//   meanNCherPhot = (2.*pi/137.)*charge*charge* 
	//                     ( 1. - 1./(nMedium*nMedium*beta*beta) )*
	//                     photEnSpectrDL*stepl;


	//     NCherPhot = meanNCherPhot;
	// Poisson:
	//     long poissNCherPhot = RandPoisson::shoot(meanNCherPhot);

	G4int poissNCherPhot = (G4int) G4Poisson(meanNCherPhot);

	if(poissNCherPhot < 0) poissNCherPhot = 0; 

	NCherPhot = poissNCherPhot * effPMTandTransport * d_qz;



#ifdef debug
	LogDebug("ForwardSim") << " ==============================> start all "
			       << "information:<========= \n" << " =====> for "
			       << "test:<===  \n" << " variant = " << variant  
			       << "\n thgrad = " << thgrad  << "\n thchergrad "
			       << "= " << thchergrad  << "\n DelFibPartgrad = "
			       << DelFibPartgrad << "\n d_qz = " << d_qz  
			       << "\n =====> Start Step Information <===  \n"
			       << " ===> calo preStepPoint info <===  \n" 
			       << " hitPoint = " << hitPoint  << "\n"
			       << " hitMom = " << hit_mom  << "\n"
			       << " stepControlFlag = " << stepControlFlag 
	  // << "\n curprocess = " << curprocess << "\n"
	  // << " nameProcess = " << nameProcess 
			       << "\n charge = " << charge << "\n"
			       << " beta = " << beta << "\n"
			       << " bThreshold = " << bThreshold << "\n"
			       << " thgrad =" << thgrad << "\n"
			       << " effPMTandTransport=" << effPMTandTransport 
	  // << "\n volume = " << name 
			       << "\n nameVolume = " << nameVolume << "\n"
			       << " nMedium = " << nMedium << "\n"
	  //  << " rad length = " << rad << "\n"
	  //  << " material = " << mat << "\n"
			       << " stepl = " << stepl << "\n"
			       << " photEnSpectrDE = " << photEnSpectrDE <<"\n"
			       << " edep = " << edep << "\n"
			       << " ===> calo theTrack info <=== " << "\n"
			       << " particleType = " << particleType << "\n"
			       << " primaryID = " << primaryID << "\n"
			       << " entot= " << entot << "\n"
			       << " vert_eta= " << eta  << "\n"
			       << " vert_phi= " << phi << "\n"
			       << " vert_mom= " << vert_mom  << "\n"
			       << " ===> calo hit preStepPointinfo <=== "<<"\n"
			       << " local point = " << localPoint << "\n"
			       << " ==============================> final info"
			       << ":  <=== \n" 
			       << " meanNCherPhot = " << meanNCherPhot << "\n"
			       << " poissNCherPhot = " << poissNCherPhot <<"\n"
			       << " NCherPhot = " << NCherPhot;
#endif 
	
      }
    }


#ifdef debug
    LogDebug("ForwardSim") << "CastorSD:: " << nameVolume 
      //      << " Light Collection Efficiency " << weight
			   << " Weighted Energy Deposit " << edep/MeV 
			   << " MeV\n";
#endif
    return NCherPhot;
  } 
}

uint32_t CastorSD::setDetUnitId(G4Step* aStep) {
  return (numberingScheme == 0 ? 0 : numberingScheme->getUnitID(aStep));
}

void CastorSD::setNumberingScheme(CastorNumberingScheme* scheme) {

  if (scheme != 0) {
    edm::LogWarning("ForwardSim") << "CastorSD: updates numbering scheme for " 
				  << GetName();
    if (numberingScheme) delete numberingScheme;
    numberingScheme = scheme;
  }
}

double CastorSD::curve_Castor(G4String& nameVolume, G4StepPoint* stepPoint) {

  // take into account light collection curve for sensitive plate
  //                                 it is case - no fibres!!!

  double weight = 1.;
  G4ThreeVector  localPoint = setToLocal(stepPoint->GetPosition(),
					 stepPoint->GetTouchable());
  double crlength = 230.;
  if (nameVolume == "CASF" || nameVolume == "C3TF" || nameVolume == "C4TF") 
    crlength = 220.;
  double dapd = 0.5 * crlength - localPoint.z();
  if (dapd >= -0.1 || dapd <= crlength+0.1) {
    if (dapd <= 100.)
      weight = 1.05 - dapd * 0.0005;
  } else {
    edm::LogInfo("ForwardSim") << "CastorSD, light collection curve: wrong "
			       << "distance " << dapd << " crlength = " 
			       << crlength << " crystal name = " << nameVolume 
			       << " z of localPoint = " << localPoint.z() 
			       << " take weight = " << weight;
  }
#ifdef debug
  LogDebug("ForwardSim") << "CastorSD, light collection curve : " << dapd 
			 << " crlength = " << crlength << " crystal name = " 
			 << nameVolume << " z of localPoint = " 
			 << localPoint.z() << " take weight = " << weight;
#endif
  return weight;
}
