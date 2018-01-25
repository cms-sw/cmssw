///////////////////////////////////////////////////////////////////////////////
// File: CastorSD.cc
// Date: 02.04
// UpDate: 07.04 - C3TF & C4TF semi-trapezoid added
// Description: Sensitive Detector class for Castor
///////////////////////////////////////////////////////////////////////////////

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/TrackInformationExtractor.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"

#include "SimG4CMS/Forward/interface/CastorSD.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#include "G4ios.hh"
#include "G4Cerenkov.hh"
#include "G4LogicalVolumeStore.hh"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "Randomize.hh"
#include "G4Poisson.hh"

//#define debugLog

CastorSD::CastorSD(const std::string& name, const DDCompactView & cpv,
                   const SensitiveDetectorCatalog & clg,
                   edm::ParameterSet const & p, 
                   const SimTrackManager* manager) : 
  CaloSD(name, cpv, clg, p, manager), numberingScheme(nullptr), lvC3EF(nullptr),
  lvC3HF(nullptr), lvC4EF(nullptr), lvC4HF(nullptr), lvCAST(nullptr) {
  
  edm::ParameterSet m_CastorSD = p.getParameter<edm::ParameterSet>("CastorSD");
  useShowerLibrary  = m_CastorSD.getParameter<bool>("useShowerLibrary");
  energyThresholdSL = m_CastorSD.getParameter<double>("minEnergyInGeVforUsingSLibrary");
  energyThresholdSL = energyThresholdSL*GeV;   //  Convert GeV => MeV 
          
  non_compensation_factor = m_CastorSD.getParameter<double>("nonCompensationFactor");
  
  if (useShowerLibrary) {
    showerLibrary = new CastorShowerLibrary(name, p);
    setParameterized(true);
  }
  setNumberingScheme(new CastorNumberingScheme());
  
  edm::LogInfo("ForwardSim") 
    << "********************************************************\n"
    << "* Constructing a CastorSD  with name " << GetName() << "\n"
    << "* Using Castor Shower Library: " << useShowerLibrary << "\n"
    << "********************************************************";
    
  const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
  for (auto lv : *lvs) {
    if (strcmp((lv->GetName()).c_str(),"C3EF") == 0) { lvC3EF = lv; }
    if (strcmp((lv->GetName()).c_str(),"C3HF") == 0) { lvC3HF = lv; }
    if (strcmp((lv->GetName()).c_str(),"C4EF") == 0) { lvC4EF = lv; }
    if (strcmp((lv->GetName()).c_str(),"C4HF") == 0) { lvC4HF = lv; }
    if (strcmp((lv->GetName()).c_str(),"CAST") == 0) { lvCAST = lv; }
    if (lvC3EF != nullptr && lvC3HF != nullptr && lvC4EF != nullptr && 
        lvC4HF != nullptr && lvCAST != nullptr) { break; }
  }
  edm::LogInfo("ForwardSim") << "CastorSD:: LogicalVolume pointers\n"
                             << lvC3EF << " for C3EF; " << lvC3HF 
                             << " for C3HF; " << lvC4EF << " for C4EF; " 
                             << lvC4HF << " for C4HF; " 
                             << lvCAST << " for CAST. ";
}

//=============================================================================================

CastorSD::~CastorSD() {
  delete showerLibrary;
}

//=============================================================================================

void CastorSD::initRun(){
  if (useShowerLibrary) {
    G4ParticleTable * theParticleTable = G4ParticleTable::GetParticleTable();
    showerLibrary->initParticleTable(theParticleTable);
  }
}

//=============================================================================================

double CastorSD::getEnergyDeposit(const G4Step * aStep) {
  
  double NCherPhot = 0.;

  // Get theTrack 
  auto const theTrack = aStep->GetTrack();
  G4int parCode = theTrack->GetDefinition()->GetPDGEncoding();

  // preStepPoint information *********************************************
  
  auto const preStepPoint = aStep->GetPreStepPoint();
  auto const currentPV    = preStepPoint->GetPhysicalVolume();
  auto const currentLV    = currentPV->GetLogicalVolume();

  const G4ThreeVector&  hit_mom  = preStepPoint->GetMomentumDirection();
  const G4ThreeVector&  hitPoint = preStepPoint->GetPosition();        
  double zint = hitPoint.z();

#ifdef debugLog
  edm::LogInfo("ForwardSim") << "CastorSD::getEnergyDeposit:"
                             << "\n CurrentStepNumber , TrackID , ParentID, Particle , VertexPosition ,"
                             << " LogicalVolumeAtVertex , PV, Time"
                             << "\n  TRACKINFO: " 
                             << theTrack->GetCurrentStepNumber() 
                             << " , " 
                             << theTrack->GetTrackID() 
                             << " , " 
                             << theTrack->GetParentID() 
                             << " , "
                             << theTrack->GetDefinition()->GetParticleName() 
                             << " , "
                             << theTrack->GetVertexPosition() 
                             << " , "
                             << theTrack->GetLogicalVolumeAtVertex()->GetName() 
                             << " , "
                             << currentPV->GetName()
                             << " , " 
                             << theTrack->GetGlobalTime();
  }
#endif

  // Full - Simulation starts here

  double meanNCherPhot=0;
    
  // remember primary particle hitting the CASTOR detector
    
  TrackInformationExtractor TIextractor;
  TrackInformation& trkInfo = TIextractor(theTrack);
  if (!trkInfo.hasCastorHit()) {
    trkInfo.setCastorHitPID(parCode);
  }
  int castorHitPID = std::abs(trkInfo.getCastorHitPID());
  
  // Check whether castor hit track is HAD
  const bool isHad = !(castorHitPID == 11 || castorHitPID == 13 || castorHitPID == 22);
  
  G4double           stepl    = aStep->GetStepLength()/cm;
  G4double           beta     = preStepPoint->GetBeta();
  G4double           charge   = preStepPoint->GetCharge();
  
  // *************************************************
  // take into account light collection curve for plate
  //      double weight = curve_Castor(nameVolume, preStepPoint);
  //      double edep   = aStep->GetTotalEnergyDeposit() * weight;
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
#ifdef debugLog
  if(theTrack->GetTrackID() == 8654) 
  edm::LogInfo("ForwardSim") << "CastorSD::getEnergyDeposit: for ID=" 
                               << theTrack->GetTrackID() << " LV: " << currentLV->GetName() 
                             << " isHad: " << isHad << " pdg= " << castorHitPID
                             << " sl= " << stepl << " Edep= " << aStep->GetTotalEnergyDeposit(); 
#endif
  if (currentLV == lvC3EF || currentLV == lvC4EF || currentLV == lvC3HF ||
      currentLV == lvC4HF) {
    
    double bThreshold = 0.67;
    double nMedium = 1.4925;
    //     double photEnSpectrDL = (1./400.nm-1./700.nm)*10000000.cm/nm; /* cm-1  */
    //     double photEnSpectrDL = 10714.285714;
    
    double photEnSpectrDE = 1.24;    /* see below   */
    /*     E = 2pi*(1./137.)*(eV*cm/370.)/lambda  =     */
    /*       = 12.389184*(eV*cm)/lambda                 */
    /*     Emax = 12.389184*(eV*cm)/400nm*10-7cm/nm  = 3.01 eV     */
    /*     Emin = 12.389184*(eV*cm)/700nm*10-7cm/nm  = 1.77 eV     */
    /*     delE = Emax - Emin = 1.24 eV  --> */
    /*   */
    /* default for Castor nameVolume  == "CASF" or (C3TF & C4TF)  */
    
    double thFullRefl = 23.;  /* 23.dergee */
    double thFullReflRad = thFullRefl*pi/180.;
    
    /* default for Castor nameVolume  == "CASF" or (C3TF & C4TF)  */
    double thFibDir = 45.;  /* .dergee */
    /* for test HF geometry volumes:   
       if(nameVolume == "GF2Q" || nameVolume == "GFNQ" ||
       nameVolume == "GR2Q" || nameVolume == "GRNQ")
       thFibDir = 0.0; // .dergee
    */
    double thFibDirRad = thFibDir*pi/180.;
    
    // at which theta the point is located:
    //     double th1    = hitPoint.theta();
    
    // theta of charged particle in LabRF(hit momentum direction):
    double costh =hit_mom.z()/sqrt(hit_mom.x()*hit_mom.x()+
                                   hit_mom.y()*hit_mom.y()+
                                   hit_mom.z()*hit_mom.z());
    if (zint < 0) costh = -costh;
    double th = acos(std::min(std::max(costh,double(-1.)),double(1.)));
    
    // just in case (can do bot use):
    if (th < 0.) th += twopi;
    
    // theta of cone with Cherenkov photons w.r.t.direction of charged part.:
    double costhcher =1./(nMedium*beta);
    double thcher = acos(std::min(std::max(costhcher,double(-1.)),double(1.)));
    
    // diff thetas of charged part. and quartz direction in LabRF:
    double DelFibPart = fabs(th - thFibDirRad);
    
    // define real distances:
    double d = fabs(tan(th)-tan(thFibDirRad));   
    
    double a = tan(thFibDirRad)+tan(fabs(thFibDirRad-thFullReflRad));   
    double r = tan(th)+tan(fabs(th-thcher));   
    
    
    // define losses d_qz in cone of full reflection inside quartz direction
    double d_qz;
#ifdef debugLog
    double variant;
#endif
    if(DelFibPart > (thFullReflRad + thcher) ) {
      d_qz = 0.; 
#ifdef debugLog
      variant=0.;
#endif
    }
    // if(d > (r+a) ) {d_qz = 0.; variant=0.;}
    else {
      if((th + thcher) < (thFibDirRad+thFullReflRad) && 
         (th - thcher) > (thFibDirRad-thFullReflRad)) {
        d_qz = 1.; 
#ifdef debugLog
        variant=1.;
#endif
      }
      // if((DelFibPart + thcher) < thFullReflRad ) {d_qz = 1.; variant=1.;}
      // if((d+r) < a ) {d_qz = 1.; variant=1.;}
      else {
        if((thFibDirRad + thFullReflRad) < (th + thcher) && 
           (thFibDirRad - thFullReflRad) > (th - thcher) ) {
          // if((thcher - DelFibPart ) > thFullReflRad ) 
          // if((r-d ) > a ) 
          d_qz = 0.; 
#ifdef debugLog
          variant=2.;
#endif
        } else {
          // if((thcher + DelFibPart ) > thFullReflRad && 
          //    thcher < (DelFibPart+thFullReflRad) ) 
          //      {
#ifdef debugLog
          variant=3.;
#endif
          
          // use crossed length of circles(cone projection)
          // dC1/dC2 : 
          double arg_arcos = 0.;
          double tan_arcos = 2.*a*d;
          if(tan_arcos != 0.) arg_arcos =(r*r-a*a-d*d)/tan_arcos; 
          arg_arcos = fabs(arg_arcos);
          double th_arcos = acos(std::min(std::max(arg_arcos,double(-1.)),double(1.)));
          d_qz = fabs(th_arcos/pi/2.);
        }
      }
    }
        
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    if(charge != 0. && beta > bThreshold )  {
      
      meanNCherPhot = 370.*charge*charge*
        ( 1. - 1./(nMedium*nMedium*beta*beta) )*
        photEnSpectrDE*stepl;
      
      const double scale = isHad ? non_compensation_factor : 1.0;
      G4int poissNCherPhot = (G4int) G4Poisson(meanNCherPhot * scale);
      
      if(poissNCherPhot < 0) poissNCherPhot = 0;
      
      double effPMTandTransport = 0.19;
      double ReflPower = 0.1;
      double proba = d_qz + (1-d_qz)*ReflPower;
      NCherPhot = poissNCherPhot*effPMTandTransport*proba*0.307;
#ifdef debugLog
      if(theTrack->GetTrackID() == 8654)       
      edm::LogInfo("ForwardSim") << " Nph= " << NCherPhot << " Np= " << poissNCherPhot
                                 << " eff= " << effPMTandTransport << " pb= " << proba
                                 << " Nmean= " << meanNCherPhot
                                 << " q=" << charge << " beta=" << beta 
                                 << " nMedium= " << nMedium << " sl= " << stepl
                                 << " Nde=" << photEnSpectrDE;
#endif
    }
  }
  return NCherPhot;
}

//=======================================================================================

uint32_t CastorSD::setDetUnitId(const G4Step* aStep) {
  return (numberingScheme == nullptr ? 0 : numberingScheme->getUnitID(aStep));
}

//=======================================================================================

void CastorSD::setNumberingScheme(CastorNumberingScheme* scheme) {

  if (scheme != nullptr) {
    edm::LogInfo("ForwardSim") << "CastorSD: updates numbering scheme for " 
                               << GetName();
    delete numberingScheme;
    numberingScheme = scheme;
  }
}

//=======================================================================================

uint32_t CastorSD::rotateUnitID(uint32_t unitID, const G4Track* track, const CastorShowerEvent& shower) {
// ==============================================================
//
//   o   Exploit Castor phi symmetry to return newUnitID for  
//       shower hits based on track phi coordinate 
//
// ==============================================================
  
  // Get 'track' phi:
  double   trackPhi = track->GetPosition().phi(); 
  if(trackPhi<0) trackPhi += 2*M_PI ;
  // Get phi from primary that gave rise to SL 'shower':
  double  showerPhi = shower.getPrimPhi(); 
  if(showerPhi<0) showerPhi += 2*M_PI ;
  // Delta phi:
  
  //  Find the OctSector for which 'track' and 'shower' belong
  int  trackOctSector = (int) (  trackPhi / (M_PI/4) ) ;
  int showerOctSector = (int) ( showerPhi / (M_PI/4) ) ;
  
  uint32_t  newUnitID;
  uint32_t         sec = ( ( unitID>>4 ) & 0xF ) ;
  uint32_t  complement = ( unitID & 0xFFFFFF0F ) ;
  
  // edm::LogInfo("ForwardSim") << "\n CastorSD::rotateUnitID:  " 
  //                            << "\n      unitID = " << unitID 
  //                            << "\n         sec = " << sec 
  //                            << "\n  complement = " << complement ; 
  
  // Get 'track' z:
  double   trackZ = track->GetPosition().z();
  
  int aux ;
  int dSec = 2*(trackOctSector - showerOctSector) ;
  // if(trackZ<0)  // Good for revision 1.8 of CastorNumberingScheme
  if(trackZ>0)  // Good for revision 1.9 of CastorNumberingScheme
  {
    int sec1 = sec-dSec;
    //    sec -= dSec ;
    if(sec1<0) sec1  += 16;
    if(sec1>15) sec1 -= 16;
    sec = (uint32_t)(sec1);
  } else {
    if( dSec<0 ) sec += 16 ;
    sec += dSec ;
    aux  = (int) (sec/16) ;
    sec -= aux*16 ;
  }
  sec  = sec<<4 ;
  newUnitID = complement | sec ;
  
#ifdef debugLog
  if(newUnitID != unitID) {
    LogDebug("ForwardSim") << "\n CastorSD::rotateUnitID:  " 
                           << "\n     unitID = " << unitID 
                           << "\n  newUnitID = " << newUnitID ; 
  }
#endif
  
  return newUnitID ;

}

//=======================================================================================

bool CastorSD::getFromLibrary(const G4Step* aStep) {

/////////////////////////////////////////////////////////////////////
//
//   Method to get hits from the Shower Library
//
//   CastorShowerEvent hits returned by getShowerHits are used to  
//   replace the full simulation of the shower from theTrack
//    
//   "updateHit" save the Hits to a CaloG4Hit container
//
/////////////////////////////////////////////////////////////////////

  if (!useShowerLibrary) { return false; }
  // Get theTrack 
  auto const theTrack = aStep->GetTrack();

  // preStepPoint information *********************************************
  
  auto const preStepPoint = aStep->GetPreStepPoint();
  auto const currentPV    = preStepPoint->GetPhysicalVolume();
  auto const currentLV    = currentPV->GetLogicalVolume();

#ifdef debugLog
  if(theTrack->GetTrackID() == 2358) 
    edm::LogInfo("ForwardSim") << "CastorSD::getFromLibrary: for ID=" << theTrack->GetTrackID() 
                               << " parentID= " << theTrack->GetParentID() << " " 
                               << theTrack->GetDefinition()->GetParticleName() 
                               << " LV: " << currentLV->GetName() << " PV: " << currentPV->GetName()
                               << "\n eta= " << theTrack->GetPosition().eta() 
                               << " phi= " << theTrack->GetPosition().phi()
                               << " z(cm)= " << theTrack->GetPosition().z()/cm 
                               << " time(ns)= " << theTrack->GetGlobalTime() 
                               << " E(GeV)= " << theTrack->GetTotalEnergy()/GeV;
  } // end of if(useShowerLibrary)
#endif
  
  // if particle moves from interaction point or "backwards (halo)
  bool backward = false;
  const G4ThreeVector&  hitPoint = preStepPoint->GetPosition();        
  const G4ThreeVector&  hit_mom  = preStepPoint->GetMomentumDirection();
  double zint = hitPoint.z();
  double pz   = hit_mom.z();
  
  // Check if theTrack moves backward 
  if (pz * zint < 0.) backward = true;
  
  // Check that theTrack is above the energy threshold to use Shower Library 
  bool aboveThreshold = false;
  if(theTrack->GetKineticEnergy() > energyThresholdSL) aboveThreshold = true;
    
  // Check if theTrack is a muon (if so, DO NOT use Shower Library) 
  bool notaMuon = true;
  G4int parCode = theTrack->GetDefinition()->GetPDGEncoding();
  if (std::abs(parCode) == 13) notaMuon = false;
  
  // angle condition
  double theta_max = M_PI - 3.1305; // angle in radians corresponding to -5.2 eta
  double R_mom=sqrt(hit_mom.x()*hit_mom.x() + hit_mom.y()*hit_mom.y());
  double theta = atan2(R_mom,std::abs(pz));
  bool angleok = false;
  if ( theta < theta_max) angleok = true;
  
  // OkToUse
  double R = sqrt(hitPoint.x()*hitPoint.x() + hitPoint.y()*hitPoint.y());
  bool dot = false;
  if ( zint < -14450. && R < 45.) dot = true;
  bool inRange = true;
  if ( zint < -14700. || R > 193.) inRange = false;
  bool OkToUse = false;
  if ( inRange && !dot) OkToUse = true;
  
  const bool particleWithinShowerLibrary = aboveThreshold &&
    notaMuon && !backward && OkToUse && angleok && currentLV == lvCAST;

#ifdef debugLog
  edm::LogInfo("ForwardSim") << "CastorSD::getFromLibrary: ID= " << theTrack->GetTrackID()
                             << " E>E0 " << aboveThreshold
                             << " nMuon " << notaMuon << " backword " << backward
                             << " Ok " << OkToUse << " angle " << angleok << " LV: " 
                             << currentLV->GetName() << "  " << (currentLV == lvCAST) 
                             << " " << particleWithinShowerLibrary;
#endif
  
  // Use Castor shower library if energy is above threshold, is not a muon 
  // and is not moving backward 
  if (!particleWithinShowerLibrary) {
    
    if (currentLV == lvC3EF || currentLV == lvC4EF || currentLV == lvC3HF ||
        currentLV == lvC4HF) {
      G4double edep     = aStep->GetTotalEnergyDeposit();
      G4double beta     = preStepPoint->GetBeta();
      G4double charge   = preStepPoint->GetCharge();
      double bThreshold = 0.67;
      if(edep == 0.0 && charge != 0.0 && beta > bThreshold) { G4Poisson(0.0); }
    }
    return false;
  }
  
  // ****    Call method to retrieve hits from the ShowerLibrary   ****
  // always kill primary
  bool isKilled(true);
  CastorShowerEvent hits = showerLibrary->getShowerHits(aStep, isKilled);
  
  int primaryID = getTrackID(theTrack);

  // Reset entry point for new primary
  resetForNewPrimary(aStep);

  // Check whether track is EM or HAD
  bool isEM , isHAD ;
  if (G4TrackToParticleID::isGammaElectronPositron(theTrack)) {
    isEM = true ; isHAD = false;
  } else {
    isEM = false; isHAD = true ;
  }

#ifdef debugLog
  if(theTrack->GetTrackID() == 2358) {
  edm::LogInfo("ForwardSim") << "\n CastorSD::getFromLibrary:  " 
                             << hits.getNhit() << " hits for " << GetName() << " from " 
                             << theTrack->GetDefinition()->GetParticleName() << " of "
                             << preStepPoint->GetKineticEnergy()/GeV << " GeV and trackID " 
                             << theTrack->GetTrackID() << " isHAD: " << isHAD;
  }
#endif

  // Scale to correct energy
  double E_track = preStepPoint->GetTotalEnergy() ;
  double E_SLhit = hits.getPrimE() * GeV ;
  double scale = E_track/E_SLhit ;

  //Non compensation 
  if (isHAD){
    scale *= non_compensation_factor; // if hadronic extend the scale with the non-compensation factor
  }
  //  Loop over hits retrieved from the library
  for (unsigned int i=0; i<hits.getNhit(); ++i) {
    
    // Get nPhotoElectrons and set edepositEM / edepositHAD accordingly
    double nPhotoElectrons = hits.getNphotons(i)*scale;

    if(isEM)  {
       edepositEM  = nPhotoElectrons; 
       edepositHAD = 0.;                 
    } else if(isHAD) {
       edepositEM  = 0.;                  
       edepositHAD = nPhotoElectrons;
    }
    
    // Get hit position and time
    double                time = hits.getTime(i);
    
    // Get hit detID
    unsigned int        unitID = hits.getDetID(i);
    
    // Make the detID "rotation" from one sector to another taking into account the 
    // sectors of the impinging particle (theTrack) and of the particle that produced 
    // the 'hits' retrieved from shower library   
    unsigned int rotatedUnitID = rotateUnitID(unitID , theTrack , hits);
    currentID.setID(rotatedUnitID, time, primaryID, 0);
    processHit(aStep);
  }  //  End of loop over hits
  return true;
}

