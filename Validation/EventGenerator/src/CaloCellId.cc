/* class CaloCellId
 *
 * Simple eta-phi cell identifier, mimic calorimetric tower structure
 * phi is stored in radians 
 *
 * $Date: 2010/05/25 16:50:51 $
 * $Revision: 1.1 $
 *
 */

#include "Validation/EventGenerator/interface/CaloCellId.h"

#include "CLHEP/Units/defs.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

#include <iostream>
#include <iomanip>
#include <cmath>

CaloCellId::CaloCellId( double theEtaMin, double theEtaMax, double thePhiMin, double thePhiMax, System theSubSys ):
  etaMin(theEtaMin),etaMax(theEtaMax),phiMin(thePhiMin),phiMax(thePhiMax),subSys(theSubSys)
{}

CaloCellId::CaloCellId( const CaloCellId& id):
  etaMin(id.etaMin),etaMax(id.etaMax),phiMin(id.phiMin),phiMax(id.phiMax),subSys(id.subSys)
{}

CaloCellId::~CaloCellId() {}

bool CaloCellId::operator==(const CaloCellId& id) const {

  return (etaMin == id.etaMin && etaMax == id.etaMax && phiMin == id.phiMin && phiMax == id.phiMax && subSys == id.subSys) ? true : false ; 

}

bool CaloCellId::isInCell(double thisEta, double thisPhi){

  double myPhi = thisPhi;

  bool itIs = false;
  if ( myPhi < -1.*CLHEP::pi ) { myPhi = myPhi+CLHEP::twopi; }
  else if ( myPhi > CLHEP::pi ) { myPhi = myPhi-CLHEP::twopi; }
  if ( thisEta >= etaMin && thisEta < etaMax && myPhi >= phiMin && myPhi < phiMax ) { itIs = true; }
  return itIs;

}

double CaloCellId::getThetaCell(){

  double etaAve = 0.5*(etaMax+etaMin);
  double theta = 2.*std::atan(std::exp(-1.*etaAve));
  return theta;

}

std::ostream& operator<<(std::ostream& os, const CaloCellId& id) {
  os << "Eta range = [" 
     << std::fixed << std::setw(7) << std::setfill(' ') << std::setprecision(4) << id.getEtaMin() << "," 
     << std::fixed << std::setw(7) << std::setfill(' ') << std::setprecision(4) << id.getEtaMax() << "], Phi range = [" 
     << std::fixed << std::setw(7) << std::setfill(' ') << std::setprecision(4) << id.getPhiMin() << "," 
     << std::fixed << std::setw(7) << std::setfill(' ') << std::setprecision(4) << id.getPhiMax() << "], subsystem = " << id.getSubSys();
  return os;
}


