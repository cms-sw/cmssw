#include "SimG4CMS/CherenkovAnalysis/interface/DreamSD.h"
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

//________________________________________________________________________________________
DreamSD::DreamSD(G4String name, const DDCompactView & cpv,
	       SensitiveDetectorCatalog & clg, 
	       edm::ParameterSet const & p, const SimTrackManager* manager) : 
  CaloSD(name, cpv, clg, p, manager) {

  edm::ParameterSet m_EC = p.getParameter<edm::ParameterSet>("ECalSD");
  useBirk= m_EC.getParameter<bool>("UseBirkLaw");
  doCherenkov_ = m_EC.getParameter<bool>("doCherenkov");
  birk1  = m_EC.getParameter<double>("BirkC1")*(g/(MeV*cm2));
  birk2  = m_EC.getParameter<double>("BirkC2")*(g/(MeV*cm2))*(g/(MeV*cm2));
  slopeLY= m_EC.getParameter<double>("SlopeLightYield");
  
  edm::LogInfo("EcalSim")  << "Constructing a DreamSD  with name " << GetName() << "\n"
			   << "DreamSD:: Use of Birks law is set to      " 
			   << useBirk << "        with the two constants C1 = "
			   << birk1 << ", C2 = " << birk2 << "\n"
			   << "         Slope for Light yield is set to "
			   << slopeLY << "\n"
                           << "         Parameterization of Cherenkov is set to " 
                           << doCherenkov_;

  initMap(name,cpv);

}

//________________________________________________________________________________________
double DreamSD::getEnergyDeposit(G4Step * aStep) {

  double edep = 0;

  if ( aStep ) {
    preStepPoint        = aStep->GetPreStepPoint();
    G4String nameVolume = preStepPoint->GetPhysicalVolume()->GetName();

    // take into account light collection curve for crystals
    double weight = 1.;
    weight *= curve_LY(aStep);
    if (useBirk)   weight *= getAttenuation(aStep, birk1, birk2);
    edep    = aStep->GetTotalEnergyDeposit() * weight;
    LogDebug("EcalSim") << "DreamSD:: " << nameVolume
			<<" Light Collection Efficiency " << weight 
			<< " Weighted Energy Deposit " << edep/MeV << " MeV";

    // Get cherenkov contribution
    if ( doCherenkov_ ) {
      edep += cherenkovDeposit_( aStep );
    }

  } 

  return edep;

}


//________________________________________________________________________________________
uint32_t DreamSD::setDetUnitId(G4Step * aStep) { 

  const G4VTouchable* touch = aStep->GetPostStepPoint()->GetTouchable();
  return touch->GetReplicaNumber(0);
}


//________________________________________________________________________________________
void DreamSD::initMap(G4String sd, const DDCompactView & cpv) {

  G4String attribute = "ReadOutName";
  DDSpecificsFilter filter;
  DDValue           ddv(attribute,sd,0);
  filter.setCriteria(ddv,DDSpecificsFilter::equals);
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  fv.firstChild();

  const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
  std::vector<G4LogicalVolume *>::const_iterator lvcite;
  bool dodet=true;
  while (dodet) {
    const DDSolid & sol  = fv.logicalPart().solid();
    const std::vector<double> & paras = sol.parameters();
    G4String name = DDSplit(sol.name()).first;
    G4LogicalVolume* lv=0;
    for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) 
      if ((*lvcite)->GetName() == name) {
	lv = (*lvcite);
	break;
      }
    LogDebug("EcalSim") << "DreamSD::initMap (for " << sd << "): Solid " 
			<< name	<< " Shape " << sol.shape() <<" Parameter 0 = "
			<< paras[0] << " Logical Volume " << lv;
    double dz = 0;
    if (sol.shape() == ddbox) {
      dz = 2*paras[2];
    } else if (sol.shape() == ddtrap) {
      dz = 2*paras[0];
    }
    xtalLMap.insert(std::pair<G4LogicalVolume*,double>(lv,dz));
    dodet = fv.next();
  }
  LogDebug("EcalSim") << "DreamSD: Length Table for " << attribute << " = " 
		      << sd << ":";   
  std::map<G4LogicalVolume*,double>::const_iterator ite = xtalLMap.begin();
  int i=0;
  for (; ite != xtalLMap.end(); ite++, i++) {
    G4String name = "Unknown";
    if (ite->first != 0) name = (ite->first)->GetName();
    LogDebug("EcalSim") << " " << i << " " << ite->first << " " << name 
			<< " L = " << ite->second;
  }
}

//________________________________________________________________________________________
double DreamSD::curve_LY(G4Step* aStep) {

  G4StepPoint*     stepPoint = aStep->GetPreStepPoint();
  G4LogicalVolume* lv        = stepPoint->GetTouchable()->GetVolume(0)->GetLogicalVolume();
  G4String         nameVolume= lv->GetName();

  double weight = 1.;
  G4ThreeVector  localPoint = setToLocal(stepPoint->GetPosition(),
					 stepPoint->GetTouchable());
  double crlength = crystalLength(lv);
  double dapd = 0.5 * crlength - localPoint.z(); // Distance from closest APD
  if (dapd >= -0.1 || dapd <= crlength+0.1) {
    if (dapd <= 100.)
      weight = 1.0 + slopeLY - dapd * 0.01 * slopeLY;
  } else {
    edm::LogWarning("EcalSim") << "DreamSD: light coll curve : wrong distance "
			       << "to APD " << dapd << " crlength = " 
			       << crlength << " crystal name = " << nameVolume 
			       << " z of localPoint = " << localPoint.z() 
			       << " take weight = " << weight;
  }
  LogDebug("EcalSim") << "DreamSD, light coll curve : " << dapd 
		      << " crlength = " << crlength
		      << " crystal name = " << nameVolume 
		      << " z of localPoint = " << localPoint.z() 
		      << " take weight = " << weight;
  return weight;
}

//________________________________________________________________________________________
double DreamSD::crystalLength(G4LogicalVolume* lv) {

  double length= 230.;
  std::map<G4LogicalVolume*,double>::const_iterator ite = xtalLMap.find(lv);
  if (ite != xtalLMap.end()) length = ite->second;
  return length;
}


//________________________________________________________________________________________
// Calculate total cherenkov deposit
// Inspired by Geant4's Cherenkov implementation
double DreamSD::cherenkovDeposit_( G4Step* aStep ) {

  double cherenkovEnergy = 0;

  // Get the material and set properties if needed
  G4Material* material = aStep->GetTrack()->GetMaterial();
  G4MaterialPropertiesTable* materialPropertiesTable = material->GetMaterialPropertiesTable();
  if ( !materialPropertiesTable ) {
    if ( !setPbWO2MaterialProperties_( material ) ) {
      edm::LogWarning("EcalSim") << "Couldn't retrieve material properties table\n"
                                 << " Material = " << material->GetName();
      return cherenkovEnergy;
    }
    materialPropertiesTable = material->GetMaterialPropertiesTable();
  }

  // Retrieve refractive index
  const G4MaterialPropertyVector* Rindex = materialPropertiesTable->GetProperty("RINDEX"); 
  if ( Rindex == NULL ) {
    edm::LogWarning("EcalSim") << "Couldn't retrieve refractive index";
    return cherenkovEnergy;
  }

  LogDebug("EcalSim") << "Material properties: " << "\n"
                      << "  Pmin = " << Rindex->GetMinPhotonMomentum()
                      << "  Pmax = " << Rindex->GetMaxPhotonMomentum();
  
  // Get particle properties
  G4StepPoint* pPreStepPoint  = aStep->GetPreStepPoint();
  G4StepPoint* pPostStepPoint = aStep->GetPostStepPoint();
  const G4DynamicParticle* aParticle = aStep->GetTrack()->GetDynamicParticle();
  const double charge = aParticle->GetDefinition()->GetPDGCharge();
  // beta is averaged over step
  const double beta = 0.5*( pPreStepPoint->GetBeta() + pPostStepPoint->GetBeta() );

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
  edm::LogVerbatim("EcalSim") << "Number of photons = " << numPhotons;
  if ( numPhotons <= 0 ) {
    LogDebug("EcalSim") << "Poission number of photons is zero: " << numPhotons
                      << ", stopping here";
    return cherenkovEnergy;
  }

//   // Finally: get contribution of each photon
//   for ( int iPhoton = 0; iPhoton<numPhotons; ++iPhoton ) {
//     // Sample momentum
//     double momentum = 0., 
//     cherenkovEnergy += getPhotonEnergyDeposit_( momentum );
//   }
  

  return cherenkovEnergy;

}


//________________________________________________________________________________________
// Returns number of photons produced per GEANT-unit (millimeter) in the current medium. 
// From G4Cerenkov.cc
double DreamSD::getAverageNumberOfPhotons_( const double charge,
                                            const double beta,
                                            const G4Material* aMaterial,
                                            const G4MaterialPropertyVector* Rindex ) const
{
  const G4double rFact = 369.81/(eV * cm);

  if( beta <= 0.0 ) return 0.0;

  double BetaInverse = 1./beta;

  // Vectors used in computation of Cerenkov Angle Integral:
  // 	- Refraction Indices for the current material
  //	- new G4PhysicsOrderedFreeVector allocated to hold CAI's
 
  // Min and Max photon momenta  
  double Pmin = Rindex->GetMinPhotonMomentum();
  double Pmax = Rindex->GetMaxPhotonMomentum();

  // Min and Max Refraction Indices 
  double nMin = Rindex->GetMinProperty();	
  double nMax = Rindex->GetMaxProperty();

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
  // Interpolation is performed by the GetPhotonMomentum() and
  // GetProperty() methods of the G4MaterialPropertiesTable and
  // the GetValue() method of G4PhysicsVector.  
  else {
    Pmin = Rindex->GetPhotonMomentum(BetaInverse);
    dp = Pmax - Pmin;
    // need boolean for current implementation of G4PhysicsVector
    // ==> being phased out
    bool isOutRange;
    double CAImin = chAngleIntegrals_->GetValue(Pmin, isOutRange);
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
