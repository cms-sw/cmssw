#ifndef EcalTestBeamAlgos_EcalTBReadout_h
#define EcalTestBeamAlgos_EcalTBReadout_h

/*
 *
 * $Id: EcalTBReadout.h,v 1.2 2009/06/15 19:46:00 heltsley Exp $
 *
 */

#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

class EcalTBReadout {

public:

  // The following is not yet used, but will be the primary
  // constructor when the parameter set system is available.
  //
  EcalTBReadout(const std::string theEcalTBInfoLabel);
  ~EcalTBReadout(){};

  /// tell the readout which cells exist
  void setDetIds(const std::vector<DetId> & detIds) {theDetIds = &detIds;}

  /// search for the TT to be read
  void findTTlist(const int & crysId, const EcalTrigTowerConstituentsMap& etmap);

  /// read only the digis from the selected TT
  void readOut(EBDigiCollection & input, EBDigiCollection & output, const EcalTrigTowerConstituentsMap& etmap);

  /// read only the digis from the selected TT
  void readOut(EEDigiCollection & input, EEDigiCollection & output, const EcalTrigTowerConstituentsMap& etmap);

  /// master function to be called once per event
  void performReadout(edm::Event& event, const EcalTrigTowerConstituentsMap & theTTmap, EBDigiCollection & input, EBDigiCollection & output);

  /// master function to be called once per event
  void performReadout(edm::Event& event, const EcalTrigTowerConstituentsMap & theTTmap, EEDigiCollection & input, EEDigiCollection & output);

private:

  int theTargetCrystal_;

  std::vector<EcalTrigTowerDetId> theTTlist_;

  static const int NCRYMATRIX = 7;
  
      const std::vector<DetId>* theDetIds;

  std::string ecalTBInfoLabel_;

};

#endif 
