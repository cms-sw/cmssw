#ifndef Validation_EventGenerator_CaloCellId 
#define Validation_EventGenerator_CaloCellId

/* class CaloCellId
 *
 * Simple eta-phi cell identifier, mimic calorimetric tower structure
 * phi is stored in radians 
 *
 * $Date: 2010/05/25 16:50:50 $
 * $Revision: 1.1 $
 *
 */

#include <iostream>

class CaloCellId{

 public:

  enum System { Barrel=1,Endcap=2,Forward=3 };
  
  CaloCellId( double theEtaMin, double theEtaMax, double thePhiMin, double thePhiMax, System theSubSys );
  CaloCellId( const CaloCellId& );
  virtual ~CaloCellId();
  
  double getEtaMin() const { return etaMin; }
  double getEtaMax() const { return etaMax; }
  double getPhiMin() const { return phiMin; }
  double getPhiMax() const { return phiMax; }
  System getSubSys() const { return subSys; }

  bool operator==(const CaloCellId&) const;

  bool isInCell(double thisEta, double thisPhi);

  double getThetaCell();

 private:

  double etaMin;
  double etaMax;
  double phiMin;
  double phiMax;
  System subSys;

};

std::ostream& operator<<(std::ostream&, const CaloCellId&);
#endif

