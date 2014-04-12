///////////////////////////////////////////////////////////////////////////////
// File:  HFCherenkov.cc
// Description: Generate Cherenkov photons for HF
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFCherenkov.h"

#include "G4Poisson.hh"
#include "G4ParticleDefinition.hh"
#include "G4NavigationHistory.hh"
#include "TMath.h"

#include "Randomize.hh"
#include "G4SystemOfUnits.hh"

//#define DebugLog

HFCherenkov::HFCherenkov(edm::ParameterSet const & m_HF) {

  ref_index       = m_HF.getParameter<double>("RefIndex");
  lambda1         = ((m_HF.getParameter<double>("Lambda1"))/pow(double(10),7))*cm;
  lambda2         = ((m_HF.getParameter<double>("Lambda2"))/pow(double(10),7))*cm;
  aperture        = cos(asin(m_HF.getParameter<double>("Aperture")));
  apertureTrap    = cos(asin(m_HF.getParameter<double>("ApertureTrapped")));
  aperturetrapped = m_HF.getParameter<double>("CosApertureTrapped");
  gain            = m_HF.getParameter<double>("Gain");
  checkSurvive    = m_HF.getParameter<bool>("CheckSurvive");
  UseNewPMT       = m_HF.getParameter<bool>("UseR7600UPMT");
  sinPsimax       = m_HF.getUntrackedParameter<double>("SinPsiMax",0.5);
  fibreR          = m_HF.getUntrackedParameter<double>("FibreR",0.3)*mm;

  edm::LogInfo("HFShower") << "HFCherenkov:: initialised with ref_index " 
			   << ref_index << " lambda1/lambda2 (cm) " 
			   << lambda1/cm << "|" << lambda2/cm
			   << " aperture(total/trapped) " << aperture << "|"
			   << apertureTrap << "|" << aperturetrapped
			   << " Check photon survival in HF " << checkSurvive 
			   << " Gain " << gain << " useNewPMT " << UseNewPMT
			   << " FibreR " << fibreR;

  clearVectors();
}

HFCherenkov::~HFCherenkov() {}

int HFCherenkov::computeNPhTrapped(double pBeta, 
                                   double u, double v, double w, 
				   double step_length, double zFiber,
				   double dose, int npe_Dose) {

  if (pBeta < (1/ref_index) || step_length < 0.0001) {return 0;}

  double uv = sqrt(u*u + v*v);
  int nbOfPhotons = computeNbOfPhotons(pBeta, step_length);

  if (nbOfPhotons < 0) {
    return 0;
  } else if (nbOfPhotons > 0) {
    double w_ph=0;
    for (int i = 0; i < nbOfPhotons; i++) {
      double rand     = G4UniformRand();
      double theta_C  = acos(1./(pBeta*ref_index));
      double phi_C    = 2*M_PI*rand;
      double sinTheta = sin(theta_C);
      double cosTheta = cos(theta_C);
      double cosPhi   = cos(phi_C);
      //photon momentum
      if (uv < 0.001) { // aligned with z-axis
	w_ph = cosTheta;
      } else { // general case
	w_ph = w * cosTheta  - sinTheta * cosPhi * uv;
      }
      if (w_ph > apertureTrap) { // phton trapped inside fiber
	npe_Dose += 1; 
      }
    }
  }
  int n_photons = npe_Dose;
  return n_photons;
}

int HFCherenkov::computeNPE(G4Step * aStep, G4ParticleDefinition* pDef,
			    double pBeta, double u, double v, double w,
			    double step_length, double zFiber,
			    double dose, int npe_Dose) {

  clearVectors();
  if (!isApplicable(pDef)) {return 0;}
  if (pBeta < (1/ref_index) || step_length < 0.0001) {
#ifdef DebugLog
    LogDebug("HFShower") << "HFCherenkov::computeNPE: pBeta " << pBeta 
			 << " 1/mu " << (1/ref_index) << " step_length " 
			 << step_length;
#endif
    return 0;
  }
   
  double uv = sqrt(u*u + v*v);
  int nbOfPhotons = computeNbOfPhotons(pBeta, step_length)
    *aStep->GetTrack()->GetWeight();
#ifdef DebugLog
  LogDebug("HFShower") << "HFCherenkov::computeNPE: pBeta " << pBeta 
		       << " u/v/w " << u << "/" << v << "/" << w 
		       << " step_length " << step_length << " zFib " << zFiber 
		       << " nbOfPhotons " << nbOfPhotons;
#endif
  if (nbOfPhotons < 0) {
    return 0;
  } else if (nbOfPhotons > 0) {
    G4StepPoint * preStepPoint = aStep->GetPreStepPoint();
    G4TouchableHandle theTouchable = preStepPoint->GetTouchableHandle();
    G4ThreeVector localprepos = theTouchable->GetHistory()->
      GetTopTransform().TransformPoint(aStep->GetPreStepPoint()->GetPosition());
    G4ThreeVector localpostpos = theTouchable->GetHistory()->
      GetTopTransform().TransformPoint(aStep->GetPostStepPoint()->GetPosition());
  
    double length=sqrt((localpostpos.x()-localprepos.x())*(localpostpos.x()-localprepos.x())
                       +(localpostpos.y()-localprepos.y())*(localpostpos.y()-localprepos.y()));
    double yemit = std::sqrt(fibreR*fibreR-length*length/4.);

    double u_ph=0,v_ph=0, w_ph=0;
    for (int i = 0; i < nbOfPhotons; i++) {
      double rand     = G4UniformRand();
      double theta_C  = acos(1./(pBeta*ref_index));
      double phi_C    = 2*M_PI*rand;
      double sinTheta = sin(theta_C);
      double cosTheta = cos(theta_C);
      double cosPhi   = cos(phi_C);
      double sinPhi   = sin(phi_C);
      //photon momentum
      if (uv < 0.001) { // aligned with z-axis
	u_ph = sinTheta * cosPhi ;
	v_ph = sinTheta * sinPhi;
	w_ph = cosTheta;
      } else { // general case
	u_ph = uv * cosTheta + sinTheta * cosPhi * w;
	v_ph = sinTheta * sinPhi;
	w_ph =  w * cosTheta - sinTheta * cosPhi * uv;
      }
      double r_lambda = G4UniformRand();
      double lambda0 = (lambda1 * lambda2) / (lambda2 - r_lambda *
					      (lambda2 - lambda1));
      double lambda  = (lambda0/cm) * pow(double(10),7); // lambda is in nm
      wlini.push_back(lambda);
#ifdef DebugLog
      LogDebug("HFShower") << "HFCherenkov::computeNPE: " << i << " lambda "
			   << lambda << " w_ph " << w_ph << " aperture "
			   << aperture;
#endif
// --------------
      double xemit=length*(G4UniformRand()-0.5);
      double gam=atan2(yemit,xemit);
      double eps=atan2(v_ph,u_ph);
      double sinBeta=sin(gam-eps);
      double rho=sqrt(xemit*xemit+yemit*yemit);
      double sinEta=rho/fibreR*sinBeta;
      double cosEta=sqrt(1.-sinEta*sinEta);
      double sinPsi=sqrt(1.-w_ph*w_ph);
      double cosKsi=cosEta*sinPsi;
#ifdef DebugLog
      if (cosKsi < aperturetrapped && w_ph>0.) {
	LogDebug("HFShower") << "HFCherenkov::Trapped photon : " << u_ph << " "
			     << v_ph << " " << w_ph << " " << xemit << " "
			     << gam << " " << eps << " " << sinBeta << " "
			     << rho << " " << sinEta << " " << cosEta << " "
			     << " " << sinPsi << " " << cosKsi;
      } else {
	LogDebug("HFShower") << "HFCherenkov::Rejected photon : " << u_ph <<" "
			     << v_ph << " " << w_ph << " " << xemit << " "
			     << gam << " " << eps << " " << sinBeta << " "
			     << rho << " " << sinEta << " " << cosEta << " "
			     << " " << sinPsi << " " << cosKsi;
      }
#endif
      if (cosKsi < aperturetrapped // photon trapped inside fiber
          && w_ph>0.               // and moves to PMT
	  && sinPsi < sinPsimax) { // and is not reflected at fiber end
	wltrap.push_back(lambda);
	double prob_HF  = 1.0; //photon survived in HF
	double a0_inv   = 0.1234;  //meter^-1
	double prob_MX  = exp( - 0.5 * a0_inv ); //light mixer
	if (checkSurvive) {
	  double a_inv = a0_inv + 0.14 * pow(dose,0.30);
	  double z_meters = zFiber;
	  prob_HF  = exp(-z_meters * a_inv ); //photon survived in HF
	}
	rand = G4UniformRand();
#ifdef DebugLog
	LogDebug("HFShower") << "HFCherenkov::computeNPE: probHF " << prob_HF
			     << " prob_MX " << prob_MX << " Random " << rand 
			     << " Survive? " << (rand < (prob_HF * prob_MX));
#endif
	if (rand < (prob_HF * prob_MX)) { // survived and sent to light mixer
	  wlatten.push_back(lambda);
	  rand = G4UniformRand();
	  double effHEM = computeHEMEff(lambda);
#ifdef DebugLog
	  LogDebug("HFShower") << "HFCherenkov::computeNPE: w_ph " << w_ph 
			       << " effHEM " << effHEM << " Random " << rand 
			       << " Survive? " << (w_ph>0.997||(rand<effHEM));
#endif
	  if (w_ph>0.997 || (rand<effHEM)) { // survived HEM
	    wlhem.push_back(lambda);
	    double qEffic = computeQEff(lambda);
	    rand = G4UniformRand();
#ifdef DebugLog
	    LogDebug("HFShower") << "HFCherenkov::computeNPE: qEffic "
				 << qEffic << " Random " << rand
				 << " Survive? " <<(rand < qEffic);
#endif
	    if (rand < qEffic) { // made photoelectron
	      npe_Dose += 1;
	      momZ.push_back(w_ph);
	      wl.push_back(lambda);
	      wlqeff.push_back(lambda);
	    } // made pe
	  } // passed HEM
	} // passed fiber
      } // end of  if(w_ph < w_aperture), trapped inside fiber
    }// end of ++NbOfPhotons
  } // end of if(NbOfPhotons)}
  int npe =  npe_Dose; // Nb of photoelectrons
#ifdef DebugLog
  LogDebug("HFShower") << "HFCherenkov::computeNPE: npe " << npe;
#endif
  return npe;
}

int HFCherenkov::computeNPEinPMT(G4ParticleDefinition* pDef, double pBeta, 
                                 double u, double v, double w, 
                                 double step_length){
  clearVectors();
  int npe_ = 0;
  if (!isApplicable(pDef)) {return 0;}
  if (pBeta < (1/ref_index) || step_length < 0.0001) {
#ifdef DebugLog
    LogDebug("HFShower") << "HFCherenkov::computeNPEinPMT: pBeta " << pBeta 
			 << " 1/mu " << (1/ref_index) << " step_length " 
			 << step_length;
#endif
    return 0;
  }
   
  double uv = sqrt(u*u + v*v);
  int nbOfPhotons = computeNbOfPhotons(pBeta, step_length);
#ifdef DebugLog
  LogDebug("HFShower") << "HFCherenkov::computeNPEinPMT: pBeta " << pBeta 
		       << " u/v/w " << u << "/" << v << "/" << w 
		       << " step_length " << step_length  
		       << " nbOfPhotons " << nbOfPhotons;
#endif   
  if (nbOfPhotons < 0) {
    return 0;
  } else if (nbOfPhotons > 0) {
    double w_ph=0;
    for (int i = 0; i < nbOfPhotons; i++) {
      double rand     = G4UniformRand();
      double theta_C  = acos(1./(pBeta*ref_index));
      double phi_C    = 2*M_PI*rand;
      double sinTheta = sin(theta_C);
      double cosTheta = cos(theta_C);
      double cosPhi   = cos(phi_C); 
      //photon momentum
      if (uv < 0.001) { // aligned with z-axis
	w_ph = cosTheta;
      } else { // general case
	w_ph = w * cosTheta - sinTheta * cosPhi * uv;
      }
      double r_lambda = G4UniformRand();
      double lambda0 = (lambda1 * lambda2) / (lambda2 - r_lambda *
					      (lambda2 - lambda1));
      double lambda  = (lambda0/cm) * pow(double(10),7); // lambda is in nm
      wlini.push_back(lambda);
#ifdef DebugLog
      LogDebug("HFShower") << "HFCherenkov::computeNPEinPMT: " <<i <<" lambda "
			   << lambda << " w_ph " << w_ph << " aperture " 
			   << aperture;
#endif
      if (w_ph > aperture) { // phton trapped inside PMT glass
	wltrap.push_back(lambda);
	rand = G4UniformRand();
#ifdef DebugLog
	LogDebug("HFShower") << "HFCherenkov::computeNPEinPMT: Random " << rand
			     << " Survive? " << (rand < 1.);
#endif
	if (rand < 1.0) { // survived all the times and sent to photo-cathode
	  wlatten.push_back(lambda);
	  rand = G4UniformRand();
	  double qEffic = computeQEff(lambda);//Quantum efficiency of the PMT
	  rand = G4UniformRand();
#ifdef DebugLog
	  LogDebug("HFShower") << "HFCherenkov::computeNPEinPMT: qEffic " 
		               << qEffic << " Random " << rand 
			       << " Survive? " <<(rand < qEffic);
#endif
	  if (rand < qEffic) { // made photoelectron
	    npe_ += 1;
	    momZ.push_back(w_ph);
	    wl.push_back(lambda);
	    wlqeff.push_back(lambda);
	  } // made pe
	} // accepted all Cherenkov photons
      } // end of  if(w_ph < w_aperture), trapped inside glass
    }// end of ++NbOfPhotons
  } // end of if(NbOfPhotons)}
#ifdef DebugLog
  LogDebug("HFShower") << "HFCherenkov::computeNPEinPMT: npe " << npe_;
#endif
  return npe_;
}

std::vector<double>  HFCherenkov::getWLIni() {
  std::vector<double> v = wlini;
  return v;
}

std::vector<double>  HFCherenkov::getWLTrap() {
  std::vector<double> v = wltrap;
  return v;
}

std::vector<double>  HFCherenkov::getWLAtten() {
  std::vector<double> v = wlatten;
  return v;
}

std::vector<double>  HFCherenkov::getWLHEM() {
  std::vector<double> v  = wlhem;
  return v;
}

std::vector<double>  HFCherenkov::getWLQEff() {
  std::vector<double> v = wlqeff;
  return v;
}

std::vector<double>  HFCherenkov::getWL() {
  std::vector<double> v = wl;
  return v;
}

std::vector<double>  HFCherenkov::getMom() {
  std::vector<double> v = momZ;
  return v;
}

int HFCherenkov::computeNbOfPhotons(double beta, G4double stepL) {

  double pBeta = beta;
  double alpha = 0.0073;
  double step_length = stepL;
  double theta_C = acos(1./(pBeta*ref_index));
  double lambdaDiff = (1./lambda1 - 1./lambda2);
  double cherenPhPerLength = 2 * M_PI * alpha * lambdaDiff*cm;
  double d_NOfPhotons = cherenPhPerLength * sin(theta_C)*sin(theta_C) *  (step_length/cm);
  int nbOfPhotons = int(d_NOfPhotons);
#ifdef DebugLog
  LogDebug("HFShower") << "HFCherenkov::computeNbOfPhotons: StepLength " 
		       << step_length << " theta_C " << theta_C 
		       << " lambdaDiff " << lambdaDiff
		       << " cherenPhPerLength " << cherenPhPerLength 
		       << " Photons " << d_NOfPhotons << " " << nbOfPhotons;
#endif
  return nbOfPhotons;
}

double HFCherenkov::computeQEff(double wavelength) {

  double qeff(0.);
  if (UseNewPMT) {
    if (wavelength<=350) {
      qeff=2.45867*(TMath::Landau(wavelength,353.820,59.1324));
    } else if (wavelength>350 && wavelength<500) {
      qeff= 0.441989*exp(-pow((wavelength-358.371),2)/(2*pow((138.277),2)));
    } else if (wavelength>=500 && wavelength<550) {
      qeff= 0.271862*exp(-pow((wavelength-491.505),2)/(2*pow((47.0418),2)));
    } else if (wavelength>=550) {
      qeff= 0.137297*exp(-pow((wavelength-520.260),2)/(2*pow((75.5023),2)));
    }
#ifdef DebugLog
    LogDebug("HFShower") << "HFCherenkov:: for new PMT : wavelength === "
			 << wavelength << "\tqeff  ===\t" << qeff;
#endif
  } else {
    double y        = (wavelength - 275.) /180.;
    double func     = 1. / (1. + 250.*pow((y/5.),4));
    double qE_R7525 = 0.77 * y * exp(-y) * func ;
    qeff            = qE_R7525;
#ifdef DebugLog
    LogDebug("HFShower") << "HFCherenkov:: for old PMT : wavelength === "
			 << wavelength << "; qeff = " << qeff;
#endif
  }

#ifdef DebugLog
  LogDebug("HFShower") << "HFCherenkov::computeQEff: wavelength " << wavelength
		       << " y/func " << y << "/" << func << " qeff " << qeff;
#endif
  return qeff;
}

double HFCherenkov::computeHEMEff(double wavelength) {

  double hEMEff = 0;
  if (wavelength < 400.) {
    hEMEff = 0.;
  } else if (wavelength >= 400. && wavelength < 410.) {
    //hEMEff = .99 * (wavelength - 400.) / 10.;
    hEMEff = (-1322.453 / wavelength) + 4.2056;
  } else if (wavelength >= 410.) {
    hEMEff = 0.99;
    if (wavelength > 415. && wavelength < 445.) {
      //abs(wavelength - 430.) < 15.
      //hEMEff = 0.95;
      hEMEff = 0.97;
    } else if (wavelength > 550. && wavelength < 600.) {
      // abs(wavelength - 575.) < 25.)
      //hEMEff = 0.96;
      hEMEff = 0.98;
    } else if (wavelength > 565. && wavelength <= 635.) { // added later
      // abs(wavelength - 600.) < 35.)
      hEMEff = (701.7268 / wavelength) - 0.186;
    }
  }
#ifdef DebugLog
  LogDebug("HFShower") << "HFCherenkov::computeHEMEff: wavelength "
		       << wavelength << " hEMEff " << hEMEff;
#endif
  return hEMEff;
}

double HFCherenkov::smearNPE(int npe) {

  double pe = 0.;
  if (npe > 0) {
    for (int i = 0; i < npe; ++i) {
      double val =  G4Poisson(gain);
      pe += (val/gain) + 0.001*G4UniformRand();
    }
  }
#ifdef DebugLog
  LogDebug("HFShower") << "HFCherenkov::smearNPE: npe " << npe << " pe " << pe;
#endif
  return pe;
}

void HFCherenkov::clearVectors() {

  wl.clear();
  wlini.clear();
  wltrap.clear();
  wlatten.clear();
  wlhem.clear();
  wlqeff.clear();
  momZ.clear();
}

bool HFCherenkov::isApplicable(const G4ParticleDefinition* aParticleType) {
  bool tmp = (aParticleType->GetPDGCharge() != 0);
#ifdef DebugLog
  LogDebug("HFShower") << "HFCherenkov::isApplicable: aParticleType " 
		       << aParticleType->GetParticleName() << " PDGCharge " 
		       << aParticleType->GetPDGCharge() << " Result " << tmp;
#endif
  return tmp;
}
