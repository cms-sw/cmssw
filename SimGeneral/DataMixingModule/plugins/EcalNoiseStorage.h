#ifndef EcalNoiseStorage_h
#define SimEcalNoiseStorage_h

/** \class EcalNoiseStorage
 *
 * DataMixingModule is the EDProducer subclass 
 * that overlays rawdata events on top of MC,
 * using real data for pileup simulation
 * This class is used to pass noise hits from the individual
 * Ecal channels to the new digitization code.
 *
 * \author Mike Hildreth, University of Notre Dame
 *
 * \version   1st Version march 2014
 *
 ************************************************************/

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoiseSignalGenerator.h"


#include <map>
#include <vector>
#include <string>


namespace edm
{
  class EcalNoiseStorage : public CaloVNoiseSignalGenerator
    {
    public:

      EcalNoiseStorage() {};
      ~EcalNoiseStorage() {};

     /** standard constructor*/
     // explicit EcalNoiseStorage();

      /**Default destructor*/
      //virtual ~EcalNoiseStorage();

      void fillNoiseSignals() {};

    private:

    };
}//edm

#endif
