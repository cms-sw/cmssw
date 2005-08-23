#ifndef HCALTBTIMING_H
#define HCALTBTIMING_H 1

#include <string>
#include <iostream>
#include <vector>
#include "boost/cstdint.hpp"

namespace hcaltb {
  /** \class HcalTBTiming
      
  $Date: $
  $Revision: $
  \author P. Dudero - Minnesota
  */
  class HcalTBTiming {
  public:
    HcalTBTiming();

    // Getter methods
    double triggerTime()     const { return triggerTime_;     }
    double ttcL1Atime()      const { return ttcL1Atime_;      }
    double beamCoincidence() const { return beamCoincidence_; }
    double laserFlash()      const { return laserFlash_;      }
    double qiePhase()        const { return qiePhase_;        }

    int    M1Count()         const { return m1hits_.size();   }
    int    M2Count()         const { return m2hits_.size();   }
    int    M3Count()         const { return m3hits_.size();   }

    int    S1Count()         const { return s1hits_.size();   }
    int    S2Count()         const { return s2hits_.size();   }
    int    S3Count()         const { return s3hits_.size();   }
    int    S4Count()         const { return s4hits_.size();   }

    double M1Hits(int index) const { return m1hits_[index];   }
    double M2Hits(int index) const { return m2hits_[index];   }
    double M3Hits(int index) const { return m3hits_[index];   }

    double S1Hits(int index) const { return s1hits_[index];   }
    double S2Hits(int index) const { return s2hits_[index];   }
    double S3Hits(int index) const { return s3hits_[index];   }
    double S4Hits(int index) const { return s4hits_[index];   }

    // Setter methods
    void   setTimes (double trigger_time,
		     double ttc_l1a_time,
		     double beam_coincidence,
		     double laser_flash,
		     double qie_phase);

    void   setHits  (std::vector<double> m1hits,
		     std::vector<double> m2hits,
		     std::vector<double> m3hits,
		     std::vector<double> s1hits,
		     std::vector<double> s2hits,
		     std::vector<double> s3hits,
		     std::vector<double> s4hits);

  private:
    double triggerTime_;
    double ttcL1Atime_;
    double beamCoincidence_;
    double laserFlash_;
    double qiePhase_;

    std::vector<double> m1hits_;
    std::vector<double> m2hits_;
    std::vector<double> m3hits_;

    std::vector<double> s1hits_;
    std::vector<double> s2hits_;
    std::vector<double> s3hits_;
    std::vector<double> s4hits_;
  };

  std::ostream& operator<<(std::ostream& s, const HcalTBTiming& htbtmg);

}

#endif
