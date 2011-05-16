#include <algorithm>

#include "SimCalorimetry/EcalTestBeamAlgos/interface/EcalTBReadout.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "SimDataFormats/EcalTestBeam/interface/PEcalTBInfo.h"

EcalTBReadout::EcalTBReadout(const std::string theEcalTBInfoLabel) 
: ecalTBInfoLabel_(theEcalTBInfoLabel) {

  theTargetCrystal_ = -1;
  theTTlist_.reserve(1);

}

void EcalTBReadout::findTTlist(const int & crysId, const EcalTrigTowerConstituentsMap& etmap) {

  // search for the TT involved in the NCRYMATRIX x NCRYMATRIX
  // around the target crystal if a new target, otherwise use
  // the list already filled

  if ( crysId == theTargetCrystal_ ) { return; }

  theTTlist_.clear();

  /// step 1:
  /// find the correspondence between the crystal identified in module and its DetId
  /// assuming only 1 SM present
  
  EBDetId theTargetId;
  std::vector<DetId>::const_iterator idItr = theDetIds->begin();
  unsigned int ncount = 0;
  bool found = false;
  
  while  ( (ncount < theDetIds->size()) && !found ) 
  {
    EBDetId thisEBdetid(idItr->rawId());
    if (thisEBdetid.ic() == crysId) {
      theTargetId = thisEBdetid;
      found = true;
    }
    ++idItr;
    ++ncount;
  }
  if ( !found ) {
    throw cms::Exception("ObjectNotFound", "Ecal TB target crystal not found in geometry");
    return;
  }
  theTargetCrystal_ = theTargetId.ic();

  /// step 2:
  /// find the crystals in the matrix and fill the TT list

  int myEta = theTargetId.ieta();
  int myPhi = theTargetId.iphi();


  for ( int icrysEta = (myEta-(NCRYMATRIX-1)/2) ; icrysEta <= (myEta+(NCRYMATRIX-1)/2) ; ++icrysEta ) {
    for ( int icrysPhi = (myPhi-(NCRYMATRIX-1)/2) ; icrysPhi <= (myPhi+(NCRYMATRIX-1)/2) ; ++icrysPhi ) {
      
      /// loop on all the valid DetId and search for the good ones

      EBDetId thisEBdetid;

      idItr = theDetIds->begin();
      ncount = 0;
      found = false;
  
      while  ( (ncount < theDetIds->size()) && !found ) 
        {
          EBDetId myEBdetid(idItr->rawId());
          if ( (myEBdetid.ieta() == icrysEta) && (myEBdetid.iphi() == icrysPhi) ) {
            thisEBdetid = myEBdetid;
            found = true;
          }
          ++idItr;
          ++ncount;
        }

      if ( found ) {

        EcalTrigTowerDetId thisTTdetId=etmap.towerOf(thisEBdetid);

        LogDebug("EcalDigi") << "Crystal to be readout: sequential id = " << thisEBdetid.ic() << " eta = " << icrysEta << " phi = " << icrysPhi << " from TT = " << thisTTdetId;

        if ( theTTlist_.size() == 0 || ( theTTlist_.size() == 1 && theTTlist_[0] != thisTTdetId )) {
          theTTlist_.push_back(thisTTdetId);
        }
        else {
          std::vector<EcalTrigTowerDetId>::iterator ttFound = find(theTTlist_.begin(), theTTlist_.end(), thisTTdetId);
          if ( theTTlist_.size() > 1 && ttFound == theTTlist_.end() && *(theTTlist_.end()) != thisTTdetId ) { 
            theTTlist_.push_back(thisTTdetId);
          }
        }
      }

    }
  }

  edm::LogInfo("EcalDigi") << " TT to be read: ";
  for ( unsigned int i = 0 ; i < theTTlist_.size() ; ++i ) {
    edm::LogInfo("EcalDigi") << " TT " << i << " " << theTTlist_[i];
  }

}

void EcalTBReadout::readOut(EBDigiCollection & input, EBDigiCollection & output, const EcalTrigTowerConstituentsMap& etmap) {

  /*
  for(EBDigiCollection::const_iterator digiItr = input.begin();
      digiItr != input.end(); ++digiItr)
    {
      EcalTrigTowerDetId thisTTdetId=etmap.towerOf(digiItr->id());
      std::vector<EcalTrigTowerDetId>::iterator ttFound = find(theTTlist_.begin(), theTTlist_.end(), thisTTdetId);
      if ((ttFound != theTTlist_.end()) || *(theTTlist_.end()) == thisTTdetId) { 
        output.push_back(*digiItr);
      }
    }
  edm::LogInfo("EcalDigi") << "Read EB Digis: " << output.size();
  */

  for (unsigned int digis=0; digis<input.size(); ++digis){
    
    EBDataFrame ebdf = input[digis];
    
    EcalTrigTowerDetId thisTTdetId=etmap.towerOf(ebdf.id());
    std::vector<EcalTrigTowerDetId>::iterator ttFound = find(theTTlist_.begin(), theTTlist_.end(), thisTTdetId);

    if ((ttFound != theTTlist_.end()) || *(theTTlist_.end()) == thisTTdetId) {      
      output.push_back( ebdf.id() ) ;
      EBDataFrame ebdf2( output.back() );
      std::copy( ebdf.frame().begin(),
		 ebdf.frame().end(),
		 ebdf2.frame().begin() );
    }
  }
}

void 
EcalTBReadout::readOut( EEDigiCollection & input, 
			EEDigiCollection & output, 
			const EcalTrigTowerConstituentsMap& etmap) 
{
   for (unsigned int digis=0; digis<input.size(); ++digis)
   { 
      EEDataFrame eedf ( input[digis] ) ;

      EcalTrigTowerDetId thisTTdetId ( etmap.towerOf( eedf.id() ) ) ;

      std::vector<EcalTrigTowerDetId>::iterator ttFound 
	 ( find(theTTlist_.begin(), theTTlist_.end(), thisTTdetId ) ) ;

      if( ( ttFound != theTTlist_.end() ) ||
	  *(theTTlist_.end()) == thisTTdetId ) 
      {      
	 output.push_back( eedf.id() ) ;
	 EEDataFrame eedf2( output.back() ) ;
	 std::copy( eedf.frame().begin(), 
		    eedf.frame().end(),
		    eedf2.frame().begin() );
    }
  }
}

void EcalTBReadout::performReadout(edm::Event& event, const EcalTrigTowerConstituentsMap & theTTmap, EBDigiCollection & input, EBDigiCollection & output) {

  // TB readout
  // step 1: get the target crystal index

  edm::Handle<PEcalTBInfo> theEcalTBInfo;
  event.getByLabel(ecalTBInfoLabel_,theEcalTBInfo);

  int crysId = theEcalTBInfo->nCrystal();

  // step 2: update (if needed) the TT list to be read

  findTTlist(crysId, theTTmap);

  // step 3: perform the readout

  readOut(input, output, theTTmap);

}


void EcalTBReadout::performReadout(edm::Event& event, const EcalTrigTowerConstituentsMap & theTTmap, EEDigiCollection & input, EEDigiCollection & output) {

  // TB readout
  // step 1: get the target crystal index

  edm::Handle<PEcalTBInfo> theEcalTBInfo;
  event.getByLabel(ecalTBInfoLabel_,theEcalTBInfo);

  int crysId = theEcalTBInfo->nCrystal();

  // step 2: update (if needed) the TT list to be read

  findTTlist(crysId, theTTmap);

  // step 3: perform the readout

  readOut(input, output, theTTmap);

}
