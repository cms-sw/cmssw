
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4Poisson.hh"

// Histogramming
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TTree.h>

// Cherenkov
#include "SimG4CMS/CherenkovAnalysis/interface/DreamSD.h"
#include "SimG4CMS/CherenkovAnalysis/interface/PMTResponse.h"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

//________________________________________________________________________________________
DreamSD::DreamSD(G4String name, const DDCompactView & cpv,
	       const SensitiveDetectorCatalog & clg,
	       edm::ParameterSet const & p, const SimTrackManager* manager) : 
  CaloSD(name, cpv, clg, p, manager) {

  edm::ParameterSet m_EC = p.getParameter<edm::ParameterSet>("ECalSD");
  useBirk= m_EC.getParameter<bool>("UseBirkLaw");
  doCherenkov_ = m_EC.getParameter<bool>("doCherenkov");
  birk1  = m_EC.getParameter<double>("BirkC1")*(g/(MeV*cm2));
  birk2  = m_EC.getParameter<double>("BirkC2");
  birk3  = m_EC.getParameter<double>("BirkC3");
  slopeLY= m_EC.getParameter<double>("SlopeLightYield");
  readBothSide_ = m_EC.getUntrackedParameter<bool>("ReadBothSide", false);
  
  edm::LogInfo("EcalSim")  << "Constructing a DreamSD  with name " << GetName() << "\n"
			   << "DreamSD:: Use of Birks law is set to      " 
			   << useBirk << "  with three constants kB = "
			   << birk1 << ", C1 = " << birk2 << ", C2 = " 
			   << birk3 << "\n"
			   << "          Slope for Light yield is set to "
			   << slopeLY << "\n"
                           << "          Parameterization of Cherenkov is set to " 
                           << doCherenkov_ << " and readout both sides is "
			   << readBothSide_;

  initMap(name,cpv);

  // Init histogramming
  edm::Service<TFileService> tfile;

  if ( !tfile.isAvailable() )
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";

  ntuple_ = tfile->make<TTree>("tree","Cherenkov photons");
  if (doCherenkov_) {
    ntuple_->Branch("nphotons",&nphotons_,"nphotons/I");
    ntuple_->Branch("px",px_,"px[nphotons]/F");
    ntuple_->Branch("py",py_,"py[nphotons]/F");
    ntuple_->Branch("pz",pz_,"pz[nphotons]/F");
    ntuple_->Branch("x",x_,"x[nphotons]/F");
    ntuple_->Branch("y",y_,"y[nphotons]/F");
    ntuple_->Branch("z",z_,"z[nphotons]/F");
  }

}

//________________________________________________________________________________________
bool DreamSD::ProcessHits(G4Step * aStep, G4TouchableHistory *) {

  if (aStep == NULL) {
    return true;
  } else {
    side = 1;
    if (getStepInfo(aStep)) {
      if (hitExists() == false && edepositEM+edepositHAD>0.)
        currentHit = createNewHit();
      if (readBothSide_) {
	side = -1;
	getStepInfo(aStep);
	if (hitExists() == false && edepositEM+edepositHAD>0.)
	  currentHit = createNewHit();
      }
    }
  }
  return true;
}


//________________________________________________________________________________________
bool DreamSD::getStepInfo(G4Step* aStep) {

  preStepPoint = aStep->GetPreStepPoint();
  theTrack     = aStep->GetTrack();
  G4String nameVolume = preStepPoint->GetPhysicalVolume()->GetName();

  // take into account light collection curve for crystals
  double weight = 1.;
  weight *= curve_LY(aStep, side);
  if (useBirk)   weight *= getAttenuation(aStep, birk1, birk2, birk3);
  edepositEM  = aStep->GetTotalEnergyDeposit() * weight;
  LogDebug("EcalSim") << "DreamSD:: " << nameVolume << " Side " << side
		      <<" Light Collection Efficiency " << weight 
		      << " Weighted Energy Deposit " << edepositEM/MeV 
		      << " MeV";
  // Get cherenkov contribution
  if ( doCherenkov_ ) {
    edepositHAD = cherenkovDeposit_( aStep );
  } else {
    edepositHAD = 0;
  }

  double       time  = (aStep->GetPostStepPoint()->GetGlobalTime())/nanosecond;
  unsigned int unitID= setDetUnitId(aStep);
  if (side < 0) unitID++;
  TrackInformation * trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
  int      primaryID;

  if (trkInfo)
    primaryID = trkInfo->getIDonCaloSurface();
  else
    primaryID = 0;

  if (primaryID == 0) {
    edm::LogWarning("EcalSim") << "CaloSD: Problem with primaryID **** set by "
                               << "force to TkID **** "
                               << theTrack->GetTrackID() << " in Volume "
                               << preStepPoint->GetTouchable()->GetVolume(0)->GetName();
    primaryID = theTrack->GetTrackID();
  }

  bool flag = (unitID > 0);
  G4TouchableHistory* touch =(G4TouchableHistory*)(theTrack->GetTouchable());
  if (flag) {
    currentID.setID(unitID, time, primaryID, 0);

    LogDebug("EcalSim") << "CaloSD:: GetStepInfo for"
                        << " PV "     << touch->GetVolume(0)->GetName()
                        << " PVid = " << touch->GetReplicaNumber(0)
                        << " MVid = " << touch->GetReplicaNumber(1)
                        << " Unit   " << currentID.unitID()
                        << " Edeposit = " << edepositEM << " " << edepositHAD;
  } else {
    LogDebug("EcalSim") << "CaloSD:: GetStepInfo for"
                        << " PV "     << touch->GetVolume(0)->GetName()
                        << " PVid = " << touch->GetReplicaNumber(0)
                        << " MVid = " << touch->GetReplicaNumber(1)
                        << " Unit   " << std::hex << unitID << std::dec
                        << " Edeposit = " << edepositEM << " " << edepositHAD;
  }
  return flag;

}


//________________________________________________________________________________________
void DreamSD::initRun() {

  // Get the material and set properties if needed
  DimensionMap::const_iterator ite = xtalLMap.begin();
  G4LogicalVolume* lv = (ite->first);
  G4Material* material = lv->GetMaterial();
  edm::LogInfo("EcalSim") << "DreamSD::initRun: Initializes for material " 
			  << material->GetName() << " in " << lv->GetName();
  materialPropertiesTable = material->GetMaterialPropertiesTable();
  if ( !materialPropertiesTable ) {
    if ( !setPbWO2MaterialProperties_( material ) ) {
      edm::LogWarning("EcalSim") << "Couldn't retrieve material properties table\n"
                                 << " Material = " << material->GetName();
    }
    materialPropertiesTable = material->GetMaterialPropertiesTable();
  }
}


//________________________________________________________________________________________
uint32_t DreamSD::setDetUnitId(G4Step * aStep) { 
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  uint32_t id = (touch->GetReplicaNumber(1))*10 + (touch->GetReplicaNumber(0));
  LogDebug("EcalSim") << "DreamSD:: ID " << id;
  return id;
}


//________________________________________________________________________________________
void DreamSD::initMap(G4String sd, const DDCompactView & cpv) {

  G4String attribute = "ReadOutName";
  DDSpecificsMatchesValueFilter filter{DDValue(attribute,sd,0)};
  DDFilteredView fv(cpv,filter);
  fv.firstChild();

  const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
  std::vector<G4LogicalVolume *>::const_iterator lvcite;
  bool dodet=true;
  while (dodet) {
    const DDSolid & sol  = fv.logicalPart().solid();
    std::vector<double> paras(sol.parameters());
    G4String name = sol.name().name();
    G4LogicalVolume* lv=0;
    for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) 
      if ((*lvcite)->GetName() == name) {
	lv = (*lvcite);
	break;
      }
    LogDebug("EcalSim") << "DreamSD::initMap (for " << sd << "): Solid " 
			<< name	<< " Shape " << sol.shape() <<" Parameter 0 = "
			<< paras[0] << " Logical Volume " << lv;
    double length = 0, width = 0;
    // Set length to be the largest size, width the smallest
    std::sort( paras.begin(), paras.end() );
    length = 2.0*paras.back();
    width  = 2.0*paras.front();
    xtalLMap.insert( std::pair<G4LogicalVolume*,Doubles>(lv,Doubles(length,width)) );
    dodet = fv.next();
  }
  LogDebug("EcalSim") << "DreamSD: Length Table for " << attribute << " = " 
		      << sd << ":";   
  DimensionMap::const_iterator ite = xtalLMap.begin();
  int i=0;
  for (; ite != xtalLMap.end(); ite++, i++) {
    G4String name = "Unknown";
    if (ite->first != 0) name = (ite->first)->GetName();
    LogDebug("EcalSim") << " " << i << " " << ite->first << " " << name 
			<< " L = " << ite->second.first
                        << " W = " << ite->second.second;
  }
}

//________________________________________________________________________________________
double DreamSD::curve_LY(G4Step* aStep, int flag) {

  G4StepPoint*     stepPoint = aStep->GetPreStepPoint();
  G4LogicalVolume* lv        = stepPoint->GetTouchable()->GetVolume(0)->GetLogicalVolume();
  G4String         nameVolume= lv->GetName();

  double weight = 1.;
  G4ThreeVector  localPoint = setToLocal(stepPoint->GetPosition(),
					 stepPoint->GetTouchable());
  double crlength = crystalLength(lv);
  double localz   = localPoint.x();
  double dapd = 0.5 * crlength - flag*localz; // Distance from closest APD
  if (dapd >= -0.1 || dapd <= crlength+0.1) {
    if (dapd <= 100.)
      weight = 1.0 + slopeLY - dapd * 0.01 * slopeLY;
  } else {
    edm::LogWarning("EcalSim") << "DreamSD: light coll curve : wrong distance "
			       << "to APD " << dapd << " crlength = " 
			       << crlength << " crystal name = " << nameVolume 
			       << " z of localPoint = " << localz
			       << " take weight = " << weight;
  }
  LogDebug("EcalSim") << "DreamSD, light coll curve : " << dapd 
		      << " crlength = " << crlength
		      << " crystal name = " << nameVolume 
		      << " z of localPoint = " << localz
		      << " take weight = " << weight;
  return weight;
}

//________________________________________________________________________________________
const double DreamSD::crystalLength(G4LogicalVolume* lv) const {

  double length= -1.;
  DimensionMap::const_iterator ite = xtalLMap.find(lv);
  if (ite != xtalLMap.end()) length = ite->second.first;
  return length;

}

//________________________________________________________________________________________
const double DreamSD::crystalWidth(G4LogicalVolume* lv) const {

  double width= -1.;
  DimensionMap::const_iterator ite = xtalLMap.find(lv);
  if (ite != xtalLMap.end()) width = ite->second.second;
  return width;

}


//________________________________________________________________________________________
// Calculate total cherenkov deposit
// Inspired by Geant4's Cherenkov implementation
double DreamSD::cherenkovDeposit_( G4Step* aStep ) {

  double cherenkovEnergy = 0;
  if (!materialPropertiesTable) return cherenkovEnergy;
  G4Material* material = aStep->GetTrack()->GetMaterial();

  // Retrieve refractive index
  G4MaterialPropertyVector* Rindex = materialPropertiesTable->GetProperty("RINDEX"); 
  if ( Rindex == NULL ) {
    edm::LogWarning("EcalSim") << "Couldn't retrieve refractive index";
    return cherenkovEnergy;
  }

  // V.Ivanchenko - temporary close log output for 9.5
  // Material refraction properties
  int Rlength = Rindex->GetVectorLength() - 1; 
  double Pmin = Rindex->Energy(0);
  double Pmax = Rindex->Energy(Rlength);
  LogDebug("EcalSim") << "Material properties: " << "\n"
                      << "  Pmin = " << Pmin
                      << "  Pmax = " << Pmax;
  
  // Get particle properties
  G4StepPoint* pPreStepPoint  = aStep->GetPreStepPoint();
  G4StepPoint* pPostStepPoint = aStep->GetPostStepPoint();
  G4ThreeVector x0 = pPreStepPoint->GetPosition();
  G4ThreeVector p0 = aStep->GetDeltaPosition().unit();
  const G4DynamicParticle* aParticle = aStep->GetTrack()->GetDynamicParticle();
  const double charge = aParticle->GetDefinition()->GetPDGCharge();
  // beta is averaged over step
  double beta = 0.5*( pPreStepPoint->GetBeta() + pPostStepPoint->GetBeta() );
  double BetaInverse = 1.0/beta;

  LogDebug("EcalSim") << "Particle properties: " << "\n"
                      << "  charge = " << charge
                      << "  beta   = " << beta;

  // Now get number of photons generated in this step
  double meanNumberOfPhotons = getAverageNumberOfPhotons_( charge, beta, material, Rindex );
  if ( meanNumberOfPhotons <= 0.0 ) { // Don't do anything
    LogDebug("EcalSim") << "Mean number of photons is zero: " << meanNumberOfPhotons
                        << ", stopping here";
    return cherenkovEnergy;
  }

  // number of photons is in unit of Geant4...
  meanNumberOfPhotons *= aStep->GetStepLength();

  // Now get a poisson distribution
  int numPhotons = static_cast<int>( G4Poisson(meanNumberOfPhotons) );
  //edm::LogVerbatim("EcalSim") << "Number of photons = " << numPhotons;
  if ( numPhotons <= 0 ) {
    LogDebug("EcalSim") << "Poission number of photons is zero: " << numPhotons
                      << ", stopping here";
    return cherenkovEnergy;
  }

  // Material refraction properties
  double dp = Pmax - Pmin;
  double maxCos = BetaInverse / (*Rindex)[Rlength]; 
  double maxSin2 = (1.0 - maxCos) * (1.0 + maxCos);

  // Finally: get contribution of each photon
  for ( int iPhoton = 0; iPhoton<numPhotons; ++iPhoton ) {

    // Determine photon momentum
    double randomNumber;
    double sampledMomentum, sampledRI; 
    double cosTheta, sin2Theta;

    // sample a momentum (not sure why this is needed!)
    do {
      randomNumber = G4UniformRand();	
      sampledMomentum = Pmin + randomNumber * dp; 
      sampledRI = Rindex->Value(sampledMomentum);
      cosTheta = BetaInverse / sampledRI;  
      
      sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta);
      randomNumber = G4UniformRand();	
      
    } while (randomNumber*maxSin2 > sin2Theta);

    // Generate random position of photon on cone surface 
    // defined by Theta 
    randomNumber = G4UniformRand();

    double phi =  twopi*randomNumber;
    double sinPhi = sin(phi);
    double cosPhi = cos(phi);

    // Create photon momentum direction vector 
    // The momentum direction is still w.r.t. the coordinate system where the primary
    // particle direction is aligned with the z axis  
    double sinTheta = sqrt(sin2Theta); 
    double px = sinTheta*cosPhi;
    double py = sinTheta*sinPhi;
    double pz = cosTheta;
    G4ThreeVector photonDirection(px, py, pz);

    // Rotate momentum direction back to global (crystal) reference system 
    photonDirection.rotateUz(p0);

    // Create photon position and momentum
    randomNumber = G4UniformRand();
    G4ThreeVector photonPosition = x0 + randomNumber * aStep->GetDeltaPosition();
    G4ThreeVector photonMomentum = sampledMomentum*photonDirection;

    // Collect energy on APD
    cherenkovEnergy += getPhotonEnergyDeposit_( photonMomentum, photonPosition, aStep );

    // Ntuple variables
    nphotons_ = numPhotons;
    px_[iPhoton] = photonMomentum.x();
    py_[iPhoton] = photonMomentum.y();
    pz_[iPhoton] = photonMomentum.z();
    x_[iPhoton] = photonPosition.x();
    y_[iPhoton] = photonPosition.y();
    z_[iPhoton] = photonPosition.z();
  }
  
  // Fill ntuple
  ntuple_->Fill();


  return cherenkovEnergy;

}


//________________________________________________________________________________________
// Returns number of photons produced per GEANT-unit (millimeter) in the current medium. 
// From G4Cerenkov.cc
double DreamSD::getAverageNumberOfPhotons_( const double charge,
					    const double beta,
					    const G4Material* aMaterial,
					    G4MaterialPropertyVector* Rindex )
{
  const G4double rFact = 369.81/(eV * cm);

  if( beta <= 0.0 ) return 0.0;

  double BetaInverse = 1./beta;

  // Vectors used in computation of Cerenkov Angle Integral:
  // 	- Refraction Indices for the current material
  //	- new G4PhysicsOrderedFreeVector allocated to hold CAI's
 
  // Min and Max photon momenta 
  int Rlength = Rindex->GetVectorLength() - 1; 
  double Pmin = Rindex->Energy(0);
  double Pmax = Rindex->Energy(Rlength);

  // Min and Max Refraction Indices 
  double nMin = (*Rindex)[0];	
  double nMax = (*Rindex)[Rlength];

  // Max Cerenkov Angle Integral 
  double CAImax = chAngleIntegrals_->GetMaxValue();

  double dp = 0., ge = 0., CAImin = 0.;

  // If n(Pmax) < 1/Beta -- no photons generated 
  if ( nMax < BetaInverse) { } 

  // otherwise if n(Pmin) >= 1/Beta -- photons generated  
  else if (nMin > BetaInverse) {
    dp = Pmax - Pmin;	
    ge = CAImax; 
  } 
  // If n(Pmin) < 1/Beta, and n(Pmax) >= 1/Beta, then
  // we need to find a P such that the value of n(P) == 1/Beta.
  // Interpolation is performed by the GetPhotonEnergy() and
  // GetProperty() methods of the G4MaterialPropertiesTable and
  // the GetValue() method of G4PhysicsVector.  
  else {
    Pmin = Rindex->Value(BetaInverse);
    dp = Pmax - Pmin;
    // need boolean for current implementation of G4PhysicsVector
    // ==> being phased out
    double CAImin = chAngleIntegrals_->Value(Pmin);
    ge = CAImax - CAImin;
    
  }

  // Calculate number of photons 
  double numPhotons = rFact * charge/eplus * charge/eplus *
    (dp - ge * BetaInverse*BetaInverse);

  LogDebug("EcalSim") << "@SUB=getAverageNumberOfPhotons" 
                      << "CAImin = " << CAImin << "\n"
                      << "CAImax = " << CAImax << "\n"
                      << "dp = " << dp << ", ge = " << ge << "\n"
                      << "numPhotons = " << numPhotons;


  
  return numPhotons;

}


//________________________________________________________________________________________
// Set lead tungstate material properties on the fly.
// Values from Ts42 detector construction
bool DreamSD::setPbWO2MaterialProperties_( G4Material* aMaterial ) {

  std::string pbWO2Name("E_PbWO4");
  if ( pbWO2Name != aMaterial->GetName() ) { // Wrong material!
    edm::LogWarning("EcalSim") << "This is not the right material: "
                               << "expecting " << pbWO2Name 
                               << ", got " << aMaterial->GetName();
    return false;
  }

  G4MaterialPropertiesTable* table = new G4MaterialPropertiesTable();

  // Refractive index as a function of photon momentum
  // FIXME: Should somehow put that in the configuration
  const int nEntries = 14;
  double PhotonEnergy[nEntries] = { 1.7712*eV,  1.8368*eV,  1.90745*eV, 1.98375*eV, 2.0664*eV, 
                                    2.15625*eV, 2.25426*eV, 2.3616*eV,  2.47968*eV, 2.61019*eV, 
                                    2.75521*eV, 2.91728*eV, 3.09961*eV, 3.30625*eV };
  double RefractiveIndex[nEntries] = { 2.17728, 2.18025, 2.18357, 2.18753, 2.19285, 
                                       2.19813, 2.20441, 2.21337, 2.22328, 2.23619, 
                                       2.25203, 2.27381, 2.30282, 2.34666 };
  
  table->AddProperty( "RINDEX", PhotonEnergy, RefractiveIndex, nEntries );
  aMaterial->SetMaterialPropertiesTable(table); // FIXME: could this leak? What does G4 do?

  // Calculate Cherenkov angle integrals: 
  // This is an ad-hoc solution (we hold it in the class, not in the material)
  chAngleIntegrals_ = 
    std::auto_ptr<G4PhysicsOrderedFreeVector>( new G4PhysicsOrderedFreeVector() );

  int index = 0;
  double currentRI = RefractiveIndex[index];
  double currentPM = PhotonEnergy[index];
  double currentCAI = 0.0;
  chAngleIntegrals_->InsertValues(currentPM, currentCAI);
  double prevPM  = currentPM;
  double prevCAI = currentCAI;
  double prevRI  = currentRI;
  while ( ++index < nEntries ) {
    currentRI = RefractiveIndex[index];
    currentPM = PhotonEnergy[index];
    currentCAI = 0.5*(1.0/(prevRI*prevRI) + 1.0/(currentRI*currentRI));
    currentCAI = prevCAI + (currentPM - prevPM) * currentCAI;

    chAngleIntegrals_->InsertValues(currentPM, currentCAI);

    prevPM  = currentPM;
    prevCAI = currentCAI;
    prevRI  = currentRI;
  }

  LogDebug("EcalSim") << "Material properties set for " << aMaterial->GetName();

  return true;

}


//________________________________________________________________________________________
// Calculate energy deposit of a photon on APD
// - simple tracing to APD position (straight line);
// - configurable reflection probability if not straight to APD;
// - APD response function
double DreamSD::getPhotonEnergyDeposit_( const G4ThreeVector& p, 
					 const G4ThreeVector& x,
					 const G4Step* aStep )
{

  double energy = 0;

  // Crystal dimensions
  
  //edm::LogVerbatim("EcalSim") << p << x;

  // 1. Check if this photon goes straight to the APD:
  //    - assume that APD is at x=xtalLength/2.0
  //    - extrapolate from x=x0 to x=xtalLength/2.0 using momentum in x-y
  
  G4StepPoint*     stepPoint = aStep->GetPreStepPoint();
  G4LogicalVolume* lv        = stepPoint->GetTouchable()->GetVolume(0)->GetLogicalVolume();
  G4String         nameVolume= lv->GetName();

  double crlength = crystalLength(lv);
  double crwidth  = crystalWidth(lv);
  double dapd = 0.5 * crlength - x.x(); // Distance from closest APD
  double y = p.y()/p.x()*dapd;

  LogDebug("EcalSim") << "Distance to APD: " << dapd
                      << " - y at APD: " << y;

  // Not straight: compute probability
  if ( fabs(y)>crwidth*0.5 ) {
    
  }

  // 2. Retrieve efficiency for this wavelength (in nm, from MeV)
  double waveLength = p.mag()*1.239e8;
  

  energy = p.mag()*PMTResponse::getEfficiency(waveLength);

  LogDebug("EcalSim") << "Wavelength: " << waveLength << " - Energy: " << energy;

  return energy;

}
