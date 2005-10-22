#ifndef CaloTDigitizer_h
#define CaloTDigitizer_h

/** Turns hits into digis.  Assumes that 
    there's an ElectroncsSim class with the
    interface Digi convert(const CaloSamples &)

*/
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHit.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"

namespace cms {
  template<class Digi, class ElectronicsSim>
  class CaloTDigitizer
  {
  public:

    CaloTDigitizer(CaloHitResponse * hitResponse, ElectronicsSim * electronicsSim,
                   const std::vector<DetId> & cellIds)
    :  theHitResponse(hitResponse),
       theElectronicsSim(electronicsSim),
       theCellIds(cellIds) {}

    /// doesn't delete the pointers passed in
    ~CaloTDigitizer() {}

    /// turns hits into digis
    /// user must delete the pointer returned
    void run(const std::vector<CaloHit> & input, std::auto_ptr<std::vector<Digi> > & output) {
      theHitResponse->run(input);
      // make a raw digi for evey cell
      for(std::vector<DetId>::const_iterator idItr = theCellIds.begin();
          idItr != theCellIds.end(); ++idItr)
      {
         Digi digi;
         CaloSamples analogSignal(theHitResponse->findSignal(*idItr));
         bool addNoise = true;
         theElectronicsSim->analogToDigital(analogSignal , digi, addNoise);
         output->push_back(digi);
      }

      // free up some memory
      theHitResponse->clear();
    }


  private:
    CaloHitResponse * theHitResponse;
    ElectronicsSim * theElectronicsSim;
    std::vector<DetId> theCellIds;
  };
}
#endif

