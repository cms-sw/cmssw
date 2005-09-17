#ifndef CaloTElectronicsSim_h
#define CaloTElectronicsSim_h

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/DetId/interface/DetId.h"
#include<iostream>

/** This class turns a CaloSamples, representing the analog
    signal input to the readout electronics, into a
    digitized data frame
 */

namespace cms {
  template<class Digi>
  class CaloTElectronicsSim {
  public:
    CaloTElectronicsSim() {}

    void run(const CaloSamples & lf, Digi & result);
      result.setSize(frame.size());
      DetId id = frame.id();
      for(unsigned tbin = 0; tbin < frame.size(); ++tbin) {
         CaloTQIESample sample( theQIESim.makeSample(id, tbin, frame[tbin]) );
         result.setSample(tbin, sample);
      }
    }

    CaloTQIESim theQIESim;
  };
}

#endif

