#ifndef SimHcalNoiseStorage_h
#define SimHcalNoiseStorage_h

/** \class HcalNoiseStorage
 *
 * DataMixingModule is the EDProducer subclass
 * that overlays rawdata events on top of MC,
 * using real data for pileup simulation
 * This class is used to pass noise hits from the individual
 * Hcal channels to the new digitization code.
 *
 * \author Mike Hildreth, University of Notre Dame
 *
 * \version   1st Version February 2009
 *
 ************************************************************/

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoiseSignalGenerator.h"

#include <map>
#include <string>
#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {
  class HcalNoiseStorage : public CaloVNoiseSignalGenerator {
  public:
    HcalNoiseStorage(){};
    ~HcalNoiseStorage() override{};

    /** standard constructor*/
    // explicit HcalNoiseStorage();

    /**Default destructor*/
    // virtual ~HcalNoiseStorage();

    void fillNoiseSignals(CLHEP::HepRandomEngine *) override{};

  private:
  };
}  // namespace edm

#endif  // SimHcalNoiseStorage_h
