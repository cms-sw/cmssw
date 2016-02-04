// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     PHcalTB06Info
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Sunanda Banerjee
//         Created:  Tue Oct 10 10:25:44 CEST 2006
//

// system include files

// user include files
#include "SimDataFormats/HcalTestBeam/interface/PHcalTB06Info.h"
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
PHcalTB06Info::PHcalTB06Info() {
  clear();
}

// PHcalTB06Info::PHcalTB06Info(const PHcalTB06Info& rhs) {
//    // do actual copying here;
// }

PHcalTB06Info::~PHcalTB06Info() {
}

//
// assignment operators
//
// const PHcalTB06Info& PHcalTB06Info::operator=(const PHcalTB06Info& rhs) {
//   //An exception safe implementation is
//   PHcalTB06Info temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

void PHcalTB06Info::clear() {
  nPrimary =  idBeam =0;
  eBeam = etaBeam = phiBeam = 0;

  simEtot = simEEc = simEHc = 0;

  hit = 0;
  hits.clear();

  v1EvNum = v1Type = v1Nsec = 0;
  v1X = v1Y = v1Z = v1U = v1V = v1W = 0;
  v1Px = v1Py = v1Pz = 0;
  v1Sec.clear();
}

void PHcalTB06Info::setPrimary(int primary, int id, double energy, double etav,
			       double phiv) {

  nPrimary = primary;
  idBeam   = id;
  eBeam    = (float)(energy);
  etaBeam  = (float)(etav);
  phiBeam  = (float)(phiv);
  LogDebug("SimHCalData") << "PHcalTB06Info::setPrimary: nPrimary " << nPrimary
			  << " partID " << idBeam << " initE " << eBeam 
			  << " eta " << etaBeam << " phi " << phiBeam;
}

void PHcalTB06Info::setEdep(double simtot, double sime, double simh) {

  simEtot = (float)simtot;
  simEEc  = (float)sime;
  simEHc  = (float)simh;

  LogDebug("SimHCalData") << "PHcalTB06Info::setEdep: simEtot " << simEtot 
			  << " simEEc " << simEEc << " simEHc " << simEHc;
}

void PHcalTB06Info::saveHit(unsigned int id, double eta, double phi, double e, 
			    double t) {

  int nh = hit;
  hit++;
  PHcalTB06Info::Hit newHit;
  newHit.id  = id;
  newHit.eta = (float)(eta);
  newHit.phi = (float)(phi);
  newHit.e   = (float)(e);
  newHit.t   = (float)(t);
  hits.push_back(newHit);
  LogDebug("SimHCalData") << "PHcalTB06Info::saveHit " << hit << " ID 0x" 
			  << std::hex << hits[nh].id << std::dec << " Eta " 
			  << hits[nh].eta << " Phi " << hits[nh].phi 
			  << " E " << hits[nh].e << " t " << hits[nh].t;
}

void PHcalTB06Info::setVtxPrim(int evNum, int type, double x, double y, 
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
  LogDebug("SimHCalData") << "PHcalTB06Info::setVtxPrim: EvNum " << v1EvNum 
			  << " Type " << v1Type << " X/Y/Z/R " << v1X << "/" 
			  << v1Y << "/" << v1Z << "/" << v1R << " Px/Py/Pz " 
			  << v1Px << "/" << v1Py << "/" << v1Pz << " U/V/W "
			  << v1U << "/" << v1V << "/" << v1W;
}

void PHcalTB06Info::setVtxSec(int id, int pdg, double px, double py, double pz,
			      double ek) {

  int ns = v1Nsec;
  v1Nsec++;
  PHcalTB06Info::Vtx newVtx;
  newVtx.trackID = id;
  newVtx.partID  = pdg;
  newVtx.px      = (float)(px);
  newVtx.py      = (float)(py);
  newVtx.pz      = (float)(pz);
  newVtx.eKin    = (float)(ek);
  v1Sec.push_back(newVtx);
  LogDebug("SimHCalData") << "PHcalTB06Info::setVtxSec " << v1Nsec << " ID " 
			  << v1Sec[ns].trackID << " PDG Code " 
			  << v1Sec[ns].partID << " Px/Py/Pz/Ek " 
			  << v1Sec[ns].px << "/" << v1Sec[ns].py << "/" 
			  << v1Sec[ns].pz << "/" << v1Sec[ns].eKin;
}
