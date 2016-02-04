/* class CaloCellManager
 *
 * Simple eta-phi cell structure manager, mimic calorimetric tower structure
 *
 * $Date: 2010/05/25 16:50:51 $
 * $Revision: 1.1 $
 *
 */

#include "Validation/EventGenerator/interface/CaloCellManager.h"

#include "CLHEP/Units/defs.h"
#include "CLHEP/Units/SystemOfUnits.h"

#include <iomanip>

CaloCellManager::CaloCellManager(const unsigned int theVerbosity):
  verbosity(theVerbosity)
{

  //initialize constants

  init();

  // build the calo cells identifiers

  builder();

}

CaloCellManager::~CaloCellManager(){

  for (unsigned int i = 0; i < theCellCollection.size(); i++) {
    delete theCellCollection[i];
  }

}

void CaloCellManager::init(){

  etaLim.reserve(nBarrelEta+nEndcapEta+nForwardEta+1);
  phiLimBar.reserve(nBarrelEta+1);
  phiLimEnd.reserve(nEndcapEta+1);
  phiLimFor.reserve(nForwardEta+1);

  // Barrel ranges

  double firstEta = 0.;
  double etaBarrelBin = 0.087;
  for (unsigned int ietabin = 0; ietabin <= nBarrelEta; ietabin++) {
    etaLim.push_back(firstEta+ietabin*etaBarrelBin);
  }

  double firstPhi = -180.;
  double phiBarrelBin = (double)360/nBarrelPhi;
  for (unsigned int iphibin = 0; iphibin <= nBarrelPhi; iphibin++) {
    phiLimBar.push_back((firstPhi+iphibin*phiBarrelBin)*CLHEP::degree);
  }

  // Endcap ranges (compromise wrt real CMS)

  firstEta = etaBarrelBin*nBarrelEta;
  double etaEndcapBin = 0.131;
  for (unsigned int ietabin = 1; ietabin <= nEndcapEta; ietabin++) {
    etaLim.push_back(firstEta+ietabin*etaEndcapBin);
  }
  double phiEndcapBin = (double)360/nEndcapPhi;
  for (unsigned int iphibin = 0; iphibin <= nEndcapPhi; iphibin++) {
    phiLimEnd.push_back((firstPhi+iphibin*phiEndcapBin)*CLHEP::degree);
  }
  
  // Forward ranges (compromise wrt real CMS)

  etaLim.push_back(3.139);
  etaLim.push_back(3.314);
  etaLim.push_back(3.489);
  etaLim.push_back(3.664);
  etaLim.push_back(3.839);
  etaLim.push_back(4.013);
  etaLim.push_back(4.191);
  etaLim.push_back(4.363);
  etaLim.push_back(4.538);
  etaLim.push_back(4.716);
  etaLim.push_back(4.889);
  etaLim.push_back(5.191);

  double phiForwardBin = (double)360/nForwardPhi;
  for (unsigned int iphibin = 0; iphibin <= nForwardPhi; iphibin++) {
    phiLimFor.push_back((firstPhi+iphibin*phiForwardBin)*CLHEP::degree);
  }

  if ( verbosity > 0 ) {
    std::cout << "Number of eta ranges = " << nBarrelEta+nEndcapEta+nForwardEta << std::endl;
    for (unsigned int i = 0; i < etaLim.size(); i++) {
      std::cout << "Eta range limit # " << i << " = " << etaLim[i] << std::endl;
    }
    for (unsigned int i = 0; i < phiLimBar.size(); i++) {
      std::cout << "Phi barrel range limit # " << i << " = " << phiLimBar[i] << std::endl;
    }
    for (unsigned int i = 0; i < phiLimEnd.size(); i++) {
      std::cout << "Phi endcap range limit # " << i << " = " << phiLimEnd[i] << std::endl;
    }
    for (unsigned int i = 0; i < phiLimFor.size(); i++) {
      std::cout << "Phi forward range limit # " << i << " = " << phiLimFor[i] << std::endl;
    }
  }

}

void CaloCellManager::builder(){

  theCellCollection.reserve(nCaloCell);

  // Barrel

  CaloCellId::System theSys = CaloCellId::Barrel;

  for (unsigned int iphi = 0; iphi < nBarrelPhi; iphi++) {
    for (unsigned int ieta = 0; ieta < nBarrelEta; ieta++) {
      CaloCellId* thisCell = new CaloCellId(etaLim[ieta],etaLim[ieta+1],phiLimBar[iphi],phiLimBar[iphi+1],theSys);
      theCellCollection.push_back(thisCell);
    }
    for (unsigned int ieta = 0; ieta < nBarrelEta; ieta++) {
      CaloCellId* thisCell = new CaloCellId(-1.*etaLim[ieta+1],-1.*etaLim[ieta],phiLimBar[iphi],phiLimBar[iphi+1],theSys);
      theCellCollection.push_back(thisCell);
    }
  }

  // Endcap

  theSys = CaloCellId::Endcap;

  for (unsigned int iphi = 0; iphi < nEndcapPhi; iphi++) {
    for (unsigned int ieta = nBarrelEta; ieta < nBarrelEta+nEndcapEta; ieta++) {
      CaloCellId* thisCell = new CaloCellId(etaLim[ieta],etaLim[ieta+1],phiLimEnd[iphi],phiLimEnd[iphi+1],theSys);
      theCellCollection.push_back(thisCell);
    }
    for (unsigned int ieta = nBarrelEta; ieta < nBarrelEta+nEndcapEta; ieta++) {
      CaloCellId* thisCell = new CaloCellId(-1.*etaLim[ieta+1],-1.*etaLim[ieta],phiLimEnd[iphi],phiLimEnd[iphi+1],theSys);
      theCellCollection.push_back(thisCell);
    }
  }
  
  // Forward

  theSys = CaloCellId::Forward;

  for (unsigned int iphi = 0; iphi < nForwardPhi; iphi++) {
    for (unsigned int ieta = nBarrelEta+nEndcapEta; ieta < nBarrelEta+nEndcapEta+nForwardEta; ieta++) {
      CaloCellId* thisCell = new CaloCellId(etaLim[ieta],etaLim[ieta+1],phiLimFor[iphi],phiLimFor[iphi+1],theSys);
      theCellCollection.push_back(thisCell);
    }
    for (unsigned int ieta = nBarrelEta+nEndcapEta; ieta < nBarrelEta+nEndcapEta+nForwardEta; ieta++) {
      CaloCellId* thisCell = new CaloCellId(-1.*etaLim[ieta+1],-1.*etaLim[ieta],phiLimFor[iphi],phiLimFor[iphi+1],theSys);
      theCellCollection.push_back(thisCell);
    }
  }

  if ( verbosity > 0 ) {
    std::cout << "Number of cells = " << nCaloCell << std::endl;
    for (unsigned int i = 0; i < theCellCollection.size(); i++) {
      std::cout << "Cell # " << std::setfill(' ') << std::setw(4) << i << " = " << *(theCellCollection[i]) << std::endl;
    }
  }

}

unsigned int CaloCellManager::getCellIndexFromAngle(double eta, double phi){

  unsigned int theIndex = 1000000;
  for ( unsigned int i = 0; i < theCellCollection.size(); i++) {
    if ( theCellCollection[i]->isInCell(eta, phi) ) { theIndex = i; continue; }
  }
  return theIndex;

}

CaloCellId* CaloCellManager::getCellFromIndex(unsigned int id){

  if ( id < theCellCollection.size() ) { return theCellCollection[id]; }
  return NULL; 

}

std::vector<double> CaloCellManager::getEtaRanges(){

  std::vector<double> theEtaRanges(etaLim); 
  return theEtaRanges;

}
