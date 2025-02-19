#ifndef Validation_EventGenerator_CaloCellManager 
#define Validation_EventGenerator_CaloCellManager

/* class CaloCellManager
 *
 * Simple eta-phi cell structure manager, mimic calorimetric tower structure
 *
 * $Date: 2010/05/25 16:50:50 $
 * $Revision: 1.1 $
 *
 */

#include "Validation/EventGenerator/interface/CaloCellId.h"

#include <vector>

class CaloCellManager{

 public:

  typedef std::vector<CaloCellId*> CaloCellCollection;

  CaloCellManager(unsigned int theVerbosity);
  virtual ~CaloCellManager();

  unsigned int getCellIndexFromAngle(double eta, double phi);
  CaloCellId* getCellFromIndex(unsigned int id);
  std::vector<double> getEtaRanges();

  // approximated CMS eta-phi calorimetri tower grid

  static const unsigned int nBarrelEta = 16;
  static const unsigned int nEndcapEta = 12;
  static const unsigned int nForwardEta = 12;

  static const unsigned int nBarrelPhi = 72;
  static const unsigned int nEndcapPhi = 36;
  static const unsigned int nForwardPhi = 36;

  static const unsigned int nBarrelCell = 2*nBarrelEta*nBarrelPhi;
  static const unsigned int nEndcapCell = 2*nEndcapEta*nEndcapPhi;
  static const unsigned int nForwardCell = 2*nForwardEta*nForwardPhi;

  static const unsigned int nCaloCell = nBarrelCell+nEndcapCell+nForwardCell;
  
 private:

  void init();
  void builder();

  unsigned int verbosity;

  std::vector<double> etaLim;
  std::vector<double> phiLimBar;
  std::vector<double> phiLimEnd;
  std::vector<double> phiLimFor;

  CaloCellCollection theCellCollection;

};

#endif
