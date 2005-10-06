#ifndef HCALTBRUNDATA_H
#define HCALTBRUNDATA_H 1

#include <string>
#include <iostream>
#include "boost/cstdint.hpp"


  /** \class HcalTBRunData
      
  $Date: 2005/08/23 01:07:18 $
  $Revision: 1.1 $
  \author P. Dudero - Minnesota
  */
  class HcalTBRunData {
  public:
    HcalTBRunData();

    // Getter methods
    const std::string& runType()       const { return runType_;       }
    const std::string& beamMode()      const { return beamMode_;      }

    double             beamEnergyGeV() const { return beamEnergyGeV_; }

    // Setter methods
    void               setRunData    ( const char *run_type,
				       const char *beam_mode,
				       double      beam_energy_gev );

  private:
    std::string runType_;
    std::string beamMode_;
    double beamEnergyGeV_;
  };

  std::ostream& operator<<(std::ostream& s, const HcalTBRunData& htbrd);

#endif
