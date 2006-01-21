#ifndef CaloSimAlgos_CaloTDigitizer_h
#define CaloSimAlgos_CaloTDigitizer_h

/** Turns hits into digis.  Assumes that 
    there's an ElectroncsSim class with the
    interface Digi convert(const CaloSamples &)

*/
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"


template<class Traits>
class CaloTDigitizer
{
public:
  /// these are the types that need to be defined in the Traits
  /// class.  The ElectronicsSim needs to have an interface
  /// that you'll see in the run() method
  typedef typename Traits::ElectronicsSim ElectronicsSim;
  typedef typename Traits::Digi Digi;
  typedef typename Traits::DigiCollection DigiCollection;

  CaloTDigitizer(CaloHitResponse * hitResponse, ElectronicsSim * electronicsSim, bool addNoise)
  :  theHitResponse(hitResponse),
     theElectronicsSim(electronicsSim),
     theDetIds(0),
     addNoise_(addNoise)
  {
  }



  /// doesn't delete the pointers passed in
  ~CaloTDigitizer() {}

  /// will get done every event by the producers
  void setDetIds(const std::vector<DetId> & detIds) {theDetIds = detIds;}

  /// turns hits into digis
  /// user must delete the pointer returned
  void run(MixCollection<PCaloHit> & input, DigiCollection & output) {
    assert(theDetIds.size() != 0);

    theHitResponse->run(input);

    theElectronicsSim->newEvent();

    // make a raw digi for evey cell
    for(std::vector<DetId>::const_iterator idItr = theDetIds.begin();
        idItr != theDetIds.end(); ++idItr)
    {
       Digi digi(*idItr);
       CaloSamples * analogSignal = theHitResponse->findSignal(*idItr);
       // don't bother digitizing if no signal and no noise
       if(analogSignal == 0 && addNoise_) {
         analogSignal = theHitResponse->makeNewSignal(*idItr);
       }
       if(analogSignal != 0) { 
         theElectronicsSim->analogToDigital(*analogSignal , digi);
         output.push_back(digi);
      }
    }

    // free up some memory
    theHitResponse->clear();
  }


private:
  CaloHitResponse * theHitResponse;
  ElectronicsSim * theElectronicsSim;
  std::vector<DetId> theDetIds;
  bool addNoise_;
};

#endif

