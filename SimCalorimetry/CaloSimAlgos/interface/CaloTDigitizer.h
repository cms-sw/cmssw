#ifndef CaloTDigitizer_h
#define CaloTDigitizer_h

/** Turns hits into digis.  Assumes that 
    there's an ElectroncsSim class with the
    interface Digi convert(const CaloSamples &)

*/
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHit.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"

namespace cms {
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

    CaloTDigitizer(CaloHitResponse * hitResponse, ElectronicsSim * electronicsSim)
    :  theHitResponse(hitResponse),
       theElectronicsSim(electronicsSim),
       theDetIds(0) {}

    /// doesn't delete the pointers passed in
    ~CaloTDigitizer() {}

    /// will get done every event by the producers
    void setDetIds(const std::vector<DetId> & detIds) {theDetIds = detIds;}

    /// turns hits into digis
    /// user must delete the pointer returned
    void run(const std::vector<CaloHit> & input, DigiCollection & output) {
      assert(theDetIds.size() != 0);

      theHitResponse->run(input);
      // make a raw digi for evey cell
      for(std::vector<DetId>::const_iterator idItr = theDetIds.begin();
          idItr != theDetIds.end(); ++idItr)
      {
         Digi digi(*idItr);
         CaloSamples analogSignal(theHitResponse->findSignal(*idItr));
         bool addNoise = true;
         theElectronicsSim->analogToDigital(analogSignal , digi, addNoise);
         output.push_back(digi);
std::cout << digi << std::endl;
      }

      // free up some memory
      theHitResponse->clear();
    }


  private:
    CaloHitResponse * theHitResponse;
    ElectronicsSim * theElectronicsSim;
    std::vector<DetId> theDetIds;
  };
}
#endif

