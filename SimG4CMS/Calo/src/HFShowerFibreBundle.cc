///////////////////////////////////////////////////////////////////////////////
// File: HFShowerFibreBundle.cc
// Description: Hits in the fibre bundles
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFShowerFibreBundle.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "G4NavigationHistory.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define DebugLog

HFShowerFibreBundle::HFShowerFibreBundle(std::string & name, 
					 const DDCompactView & cpv,
					 edm::ParameterSet const & p) {

  edm::ParameterSet m_HF1 = p.getParameter<edm::ParameterSet>("HFShowerStraightBundle");
  facTube                 = m_HF1.getParameter<double>("FactorBundle");
  cherenkov1              = new HFCherenkov(m_HF1);
  edm::ParameterSet m_HF2 = p.getParameter<edm::ParameterSet>("HFShowerConicalBundle");
  facCone                 = m_HF2.getParameter<double>("FactorBundle");
  cherenkov2              = new HFCherenkov(m_HF2);
  edm::LogInfo("HFShower") << "HFShowerFibreBundle intialized with factors: "
			   << facTube << " for the straight portion and "
			   << facCone << " for the curved portion";
  
  G4String attribute = "OnlyForHcalSimNumbering"; 
  G4String value     = "any";
  DDValue val(attribute, value, 0.0);
  DDSpecificsFilter filter0;
  filter0.setCriteria(val, DDSpecificsFilter::not_equals,
		      DDSpecificsFilter::AND, true, true);
  DDFilteredView fv0(cpv);
  fv0.addFilter(filter0);
  if (fv0.firstChild()) {
    DDsvalues_type sv0(fv0.mergedSpecifics());

    //Special Geometry parameters
    rTable   = getDDDArray("rTable",sv0);
    edm::LogInfo("HFShower") << "HFShowerFibreBundle: " << rTable.size() 
			     << " rTable (cm)";
    for (unsigned int ig=0; ig<rTable.size(); ig++)
      edm::LogInfo("HFShower") << "HFShowerFibreBundle: rTable[" << ig 
			       << "] = " << rTable[ig]/cm << " cm";
  } else {
    edm::LogError("HFShower") << "HFShowerFibreBundle: cannot get filtered "
			      << " view for " << attribute << " matching "
			      << value;
    throw cms::Exception("Unknown", "HFShowerFibreBundle")
      << "cannot match " << attribute << " to " << name <<"\n";
  }

  attribute = "Volume";
  value     = "HFPMT";
  DDSpecificsFilter filter1;
  DDValue           ddv1(attribute,value,0);
  filter1.setCriteria(ddv1,DDSpecificsFilter::equals);
  DDFilteredView fv1(cpv);
  fv1.addFilter(filter1);
  if (fv1.firstChild()) {
    DDsvalues_type sv1(fv1.mergedSpecifics());
    std::vector<double> neta;
    neta = getDDDArray("indexPMTR",sv1);
    for (unsigned int ii=0; ii<neta.size(); ii++) {
      int index = static_cast<int>(neta[ii]);
      int ir=-1, ifib=-1;
      if (index >= 0) {
	ir   = index/10; ifib = index%10;
      }
      pmtR1.push_back(ir);
      pmtFib1.push_back(ifib);
    }
    neta = getDDDArray("indexPMTL",sv1);
    for (unsigned int ii=0; ii<neta.size(); ii++) {
      int index = static_cast<int>(neta[ii]);
      int ir=-1, ifib=-1;
      if (index >= 0) {
	ir   = index/10; ifib = index%10;
      }
      pmtR2.push_back(ir);
      pmtFib2.push_back(ifib);
    }
    edm::LogInfo("HFShower") << "HFShowerFibreBundle: gets the Index matches "
			     << "for " << neta.size() << " PMTs";
    for (unsigned int ii=0; ii<neta.size(); ii++) 
      edm::LogInfo("HFShower") << "HFShowerFibreBundle: rIndexR[" << ii 
			       << "] = " << pmtR1[ii] << " fibreR[" << ii 
			       << "] = " << pmtFib1[ii] << " rIndexL[" << ii 
			       << "] = " << pmtR2[ii] << " fibreL[" << ii 
			       << "] = " << pmtFib2[ii];
  } else {
    edm::LogWarning("HFShower") << "HFShowerFibreBundle: cannot get filtered "
				<< " view for " << attribute << " matching "
				<< value;
  }
  
}

HFShowerFibreBundle::~HFShowerFibreBundle() {
  delete cherenkov1;
  delete cherenkov2;
}

double HFShowerFibreBundle::getHits(G4Step * aStep, bool type) {

  indexR = indexF = -1;

  G4StepPoint * preStepPoint  = aStep->GetPreStepPoint(); 
  const G4VTouchable* touch   = preStepPoint->GetTouchable();
  int                 boxNo   = touch->GetReplicaNumber(1);
  int                 pmtNo   = touch->GetReplicaNumber(0);
  if (boxNo <= 1) {
    indexR = pmtR1[pmtNo-1];
    indexF = pmtFib1[pmtNo-1];
  } else {
    indexR = pmtR2[pmtNo-1];
    indexF = pmtFib2[pmtNo-1];
  }

#ifdef DebugLog
  double edep = aStep->GetTotalEnergyDeposit();
  LogDebug("HFShower") << "HFShowerFibreBundle: Box " << boxNo << " PMT "
		       << pmtNo << " Mapped Indices " << indexR << ", "
		       << indexF << " Edeposit " << edep/MeV << " MeV";
#endif

  double photons = 0;
  if (indexR >= 0 && indexF > 0) {
    G4Track *aTrack = aStep->GetTrack();
    G4ParticleDefinition *particleDef = aTrack->GetDefinition();
    double stepl = aStep->GetStepLength();
    double beta  = preStepPoint->GetBeta();
    G4ThreeVector pDir = aTrack->GetDynamicParticle()->GetMomentumDirection();
    G4ThreeVector localMom = preStepPoint->GetTouchable()->GetHistory()->
      GetTopTransform().TransformAxis(pDir);
    if (type) {
      photons = facCone*cherenkov2->computeNPEinPMT(particleDef, beta,
						    localMom.x(), localMom.y(),
						    localMom.z(), stepl);
    } else {
      photons = facTube*cherenkov1->computeNPEinPMT(particleDef, beta,
						    localMom.x(), localMom.y(),
						    localMom.z(), stepl);
    }
#ifdef DebugLog
  LogDebug("HFShower") << "HFShowerFibreBundle::getHits: for particle " 
		       << particleDef->GetParticleName() << " Step " << stepl
		       << " Beta " << beta << " Direction " << pDir
		       << " Local " << localMom << " p.e. " << photons;
#endif 

  }
  return photons;
}
 
double HFShowerFibreBundle::getRadius() {
   
  double r = 0.;
  if (indexR >= 0 && indexR+1 < (int)(rTable.size()))
    r = 0.5*(rTable[indexR]+rTable[indexR+1]);
#ifdef DebugLog
  else
    LogDebug("HFShower") << "HFShowerFibreBundle::getRadius: R " << indexR
			 << " F " << indexF;
#endif
  if (indexF == 2)  r =-r;
#ifdef DebugLog
  LogDebug("HFShower") << "HFShowerFibreBundle: Radius (" << indexR << "/" 
		       << indexF << ") " << r;
#endif
  return r;
}

std::vector<double> HFShowerFibreBundle::getDDDArray(const std::string & str, 
						     const DDsvalues_type& sv){

#ifdef DebugLog
  LogDebug("HFShower") << "HFShowerFibreBundle:getDDDArray called for " << str;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef DebugLog
    LogDebug("HFShower") << value;
#endif
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nval < 2) {
      edm::LogError("HFShower") << "HFShowerFibreBundle: # of " << str 
				<< " bins " << nval << " < 2 ==> illegal";
      throw cms::Exception("Unknown", "HFShowerFibreBundle")
	<< "nval < 2 for array " << str << "\n";
    }

    return fvec;
  } else {
    edm::LogError("HFShower") <<"HFShowerFibreBundle: cannot get array " <<str;
    throw cms::Exception("Unknown", "HFShowerFibreBundle") 
      << "cannot get array " << str << "\n";
  }
}
