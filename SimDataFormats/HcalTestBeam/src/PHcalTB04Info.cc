// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     PHcalTB04Info
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Sunanda Banerjee
//         Created:  Sun May 14 10:25:44 CEST 2006
// $Id: PHcalTB04Info.cc,v 1.4 2013/04/22 22:30:15 wmtan Exp $
//

// system include files

// user include files
#include "SimDataFormats/HcalTestBeam/interface/PHcalTB04Info.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PHcalTB04Info::PHcalTB04Info() {
  clear();
}

// PHcalTB04Info::PHcalTB04Info(const PHcalTB04Info& rhs) {
//    // do actual copying here;
// }

PHcalTB04Info::~PHcalTB04Info() {
}

//
// assignment operators
//
// const PHcalTB04Info& PHcalTB04Info::operator=(const PHcalTB04Info& rhs) {
//   //An exception safe implementation is
//   PHcalTB04Info temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

void PHcalTB04Info::clear() {
  nPrimary =  idBeam =0;
  eBeam = etaBeam = phiBeam = 0;

  simEtot = simEEc = simEHc = 0;
  digEtot = digEEc = digEHc = 0;

  nCrystal = nTower = 0;

  hit = 0;

  v1EvNum = v1Type = v1Nsec = 0;
  v1X = v1Y = v1Z = v1U = v1V = v1W = 0;
  v1Px = v1Py = v1Pz = 0;
}
  
void PHcalTB04Info::setIDs(const std::vector<int>& ide, const std::vector<int>& idh) {

  nCrystal = ide.size();
  if (nCrystal > 0) {
    idEcal.reserve(nCrystal);
    esime.reserve(nCrystal);
    edige.reserve(nCrystal);
  }
  LogDebug("SimHCalData") << "PHcalTB04Info:: Called with " << nCrystal << " crystals";
  for (int i=0; i<nCrystal; i++) {
    idEcal.push_back(ide[i]);
    LogDebug("SimHCalData") << "\tIndex for " << i << " =  0x" << std::hex << idEcal[i] << std::dec;
  }

  nTower   = idh.size();
  if (nTower > 0) {
    idHcal.reserve(nTower);
    esimh.reserve(nTower);
    edigh.reserve(nTower);
  }
  LogDebug("SimHCalData") << "PHcalTB04Info:: Called with " << nTower << " HCal towers";
  for (int i=0; i<nTower; i++) {
    idHcal.push_back(idh[i]);
    LogDebug("SimHCalData") << "\tIndex for " << i << " =  0x" << std::hex << idHcal[i] << std::dec;
  }

}

void PHcalTB04Info::setPrimary(int primary, int id, double energy, double etav,
			       double phiv) {

  nPrimary = primary;
  idBeam   = id;
  eBeam    = (float)(energy);
  etaBeam  = (float)(etav);
  phiBeam  = (float)(phiv);
  LogDebug("SimHCalData") << "PHcalTB04Info::setPrimary: nPrimary " << nPrimary << " partID " << idBeam << " initE " << eBeam << " eta " << etaBeam << " phi " << phiBeam;
}

void PHcalTB04Info::setEdep(double simtot, double sime, double simh, 
			    double digtot, double dige, double digh) {

  simEtot = (float)simtot;
  simEEc  = (float)sime;
  simEHc  = (float)simh;
  digEtot = (float)digtot;
  digEEc  = (float)dige;
  digEHc  = (float)digh;

  LogDebug("SimHCalData") << "PHcalTB04Info::setEdep: simEtot " << simEtot << " simEEc " << simEEc << " simEHc " << simEHc << " digEtot " << digEtot  << " digEEc " << digEEc << " digEHc " << digEHc;
}

void PHcalTB04Info::setEdepEcal(const std::vector<double>& esim, 
				const std::vector<double>& eqie) {

  for (int i=0; i<nCrystal; i++) {
    float edep = 0;
    if (i<int(esim.size())) esime.push_back(esim[i]);
    else                    esime.push_back(edep);
    if (i<int(eqie.size())) edige.push_back(eqie[i]);
    else                    edige.push_back(edep);

    LogDebug("SimHCalData") << "PHcalTB04Info::setEdepEcal [" << i << "] Esim = " << esime[i] << " Edig = " << edige[i];
  }
}

void PHcalTB04Info::setEdepHcal(const std::vector<double>& esim, 
				const std::vector<double>& eqie) {

  for (int i=0; i<nTower; i++) {
    float edep = 0;
    if (i<int(esim.size())) esimh.push_back(esim[i]);
    else                    esimh.push_back(edep);
    if (i<int(eqie.size())) edigh.push_back(eqie[i]);
    else                    edigh.push_back(edep);

    LogDebug("SimHCalData") << "PHcalTB04Info::setEdepHcal [" << i << "] Esim = " << esimh[i] << " Edig = " << edigh[i];
  }
}

void PHcalTB04Info::setTrnsProf(const std::vector<double>& es1, 
				const std::vector<double>& eq1, 
				const std::vector<double>& es2,
				const std::vector<double>& eq2) {

  int siz = (int)(es1.size());
  if (siz > 0) {
    latsimEta.reserve(siz);
    latdigEta.reserve(siz);
    latsimPhi.reserve(siz);
    latdigPhi.reserve(siz);
    for (int i=0; i<siz; i++) {
      latsimEta.push_back((float)(es1[i]));
      latdigEta.push_back((float)(eq1[i]));
      latsimPhi.push_back((float)(es2[i]));
      latdigPhi.push_back((float)(eq2[i]));
      LogDebug("SimHCalData") << "PHcalTB04Info::setTrnsProf [" << i << "] latsimEta = " << latsimEta[i] << " latdigEta = " << latdigEta[i] << " latsimPhi = " << latsimPhi[i] << " latdigPhi = " << latdigPhi[i];
    }
  }
}

void PHcalTB04Info::setLongProf(const std::vector<double>& es, 
				const std::vector<double>& eq) {

  int siz = (int)(es.size());
  if (siz > 0) {
    longsim.reserve(siz);
    longdig.reserve(siz);
    for (int i=0; i<siz; i++) {
      longsim.push_back((float)(es[i]));
      longdig.push_back((float)(eq[i]));
      LogDebug("SimHCalData") << "PHcalTB04Info::setLongProf [" << i << "] longsim = " << longsim[i] << " longdig = " << longdig[i];
    }
  }
}

void PHcalTB04Info::saveHit(int det, int lay, int eta, int phi, double e, 
			    double t) {

  int nh = hit;
  hit++;
  detHit.push_back(det);
  layHit.push_back(lay);
  etaHit.push_back(eta);
  phiHit.push_back(phi);
  eHit.push_back((float)(e));
  tHit.push_back((float)(t));
  LogDebug("SimHCalData") << "PHcalTB04Info::saveHit " << hit << " Det " << detHit[nh] << " layer " << layHit[nh] << " Eta " << etaHit[nh] << " Phi " << phiHit[nh] << " E " << eHit[nh] << " t " << tHit[nh];
}

void PHcalTB04Info::setVtxPrim(int evNum, int type, double x, double y, 
			       double z, double u, double v, double w, 
			       double px, double py, double pz) {

  v1EvNum = evNum;
  v1Type  = type;
  double r= sqrt(x*x+y*y+z*z);
  v1X     = (float)(x);
  v1Y     = (float)(y);
  v1Z     = (float)(z);
  v1R     = (float)(r);
  v1Px    = (float)(px);
  v1Py    = (float)(py);
  v1Pz    = (float)(pz);
  v1U     = (float)(u);
  v1V     = (float)(v);
  v1W     = (float)(w);
  LogDebug("SimHCalData") << "PHcalTB04Info::setVtxPrim: EvNum " << v1EvNum << " Type " << v1Type << " X/Y/Z/R " << v1X << "/" << v1Y << "/" << v1Z << "/" << v1R << " Px/Py/Pz " << v1Px << "/" << v1Py << "/" << v1Pz << " U/V/W " << v1U << "/" << v1V << "/" << v1W;
}

void PHcalTB04Info::setVtxSec(int id, int pdg, double px, double py, double pz,
			      double ek) {

  int ns = v1Nsec;
  v1Nsec++;
  v1secTrackID.push_back(id);
  v1secPartID.push_back(pdg);
  v1secPx.push_back((float)(px));
  v1secPy.push_back((float)(py));
  v1secPz.push_back((float)(pz));
  v1secEk.push_back((float)(ek));  
  LogDebug("SimHCalData") << "PHcalTB04Info::setVtxSec " << v1Nsec << " ID " << v1secTrackID[ns] << " PDG Code " << v1secPartID[ns] << " Px/Py/Pz/Ek " << v1secPx[ns] << "/" << v1secPy[ns] << "/" << v1secPz[ns] << "/" << v1secEk[ns];
}
