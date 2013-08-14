//
// $Id: PEcalTBInfo.cc,v 1.2 2006/10/25 16:58:05 fabiocos Exp $
//

// system include files

// user include files
#include "SimDataFormats/EcalTestBeam/interface/PEcalTBInfo.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PEcalTBInfo::PEcalTBInfo() {
  clear();
}

// PEcalTBInfo::PEcalTBInfo(const PEcalTBInfo& rhs) {
//    // do actual copying here;
// }

PEcalTBInfo::~PEcalTBInfo() {
}

//
// assignment operators
//
// const PEcalTBInfo& PEcalTBInfo::operator=(const PEcalTBInfo& rhs) {
//   //An exception safe implementation is
//   PEcalTBInfo temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

void PEcalTBInfo::clear() {
  nCrystal_ = 0;

  etaBeam_ = phiBeam_ = 0.;
  dXbeam_ = dYbeam_ = 0.;

  evXbeam_ = evYbeam_ = 0.;
  phaseShift_ = 1.;
}
  
void PEcalTBInfo::setCrystal(int nCrystal) {
  nCrystal_ = nCrystal;
}

void PEcalTBInfo::setBeamDirection(double etaBeam, double phiBeam) {
  etaBeam_ = etaBeam;
  phiBeam_ = phiBeam;
}

void PEcalTBInfo::setBeamOffset(double dXbeam, double dYbeam) {
  dXbeam_ = dXbeam;
  dYbeam_ = dYbeam;
}

void PEcalTBInfo::setBeamPosition(double evXbeam, double evYbeam) {
  evXbeam_ = evXbeam;
  evYbeam_ = evYbeam;
}

void PEcalTBInfo::setPhaseShift(double phaseShift) {
  phaseShift_ = phaseShift;
}
