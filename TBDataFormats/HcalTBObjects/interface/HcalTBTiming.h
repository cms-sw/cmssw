#ifndef HCALTBTIMING_H
#define HCALTBTIMING_H 1

#include <string>
#include <iostream>
#include <vector>
#include "boost/cstdint.hpp"

  /** \class HcalTBTiming

This class contains timing information unpacked from the
Time-to-Digital Converter (TDC).
      
  $Date: 2006/08/28 16:40:39 $
  $Revision: 1.9 $
  \author P. Dudero - Minnesota
  */
  class HcalTBTiming {
  public:
    HcalTBTiming();

    // Getter methods

    /// Returns the trigger time in ns
    double triggerTime()     const { return triggerTime_;     }

    /// Returns the Level 1 Accept time in ns
    double ttcL1Atime()      const { return ttcL1Atime_;      }

    /// Returns the laser activation time in ns
    double laserFlash()      const { return laserFlash_;      }
    /// Returns the QIE phase for 2003 testbeam data (zero otherwise)
    double qiePhase()        const { return qiePhase_;        }

    /// Returns the TOF1S time (zero otherwise)
    double TOF1Stime()       const { return TOF1Stime_;       }
    /// Returns the TOF1J time (zero otherwise)
    double TOF1Jtime()       const { return TOF1Jtime_;       }
    /// Returns the TOF2S time (zero otherwise)
    double TOF2Stime()       const { return TOF2Stime_;       }
    /// Returns the TOF2J time (zero otherwise)
    double TOF2Jtime()       const { return TOF2Jtime_;       }

    /// Returns the number of hits from Beam Coincidence
    int    BeamCoincidenceCount()         const { return beamCoincidenceHits_.size();   }
    /// Returns the number of hits from muon veto scintillator M1
    int    M1Count()         const { return m1hits_.size();   }
    /// Returns the number of hits from muon veto scintillator M2
    int    M2Count()         const { return m2hits_.size();   }
    /// Returns the number of hits from muon veto scintillator M3
    int    M3Count()         const { return m3hits_.size();   }

    /// Returns the number of hits from scintillator S1, which is 12cm x 12cm.
    int    S1Count()         const { return s1hits_.size();   }
     /// Returns the number of hits from scintillator S2, which is 4cm x 4cm.
    int    S2Count()         const { return s2hits_.size();   }
    /// Returns the number of hits from scintillator S3, which is 2cm x 2cm.
    int    S3Count()         const { return s3hits_.size();   }
    /// Returns the number of hits from scintillator S4, which is 12cm x 12cm.
    int    S4Count()         const { return s4hits_.size();   }

    /// Returns the number of hits from beam halo counter up horizontal
    int    BH1Count()         const { return bh1hits_.size();   }
    /// Returns the number of hits from beam halo counter left from particle view
    int    BH2Count()         const { return bh2hits_.size();   }
    /// Returns the number of hits from beam halo counter right from particle view
    int    BH3Count()         const { return bh3hits_.size();   }
    /// Returns the number of hits from beam halo counter down horizontal
    int    BH4Count()         const { return bh4hits_.size();   }

    /// Returns the indexed hit time from Beam Coincidence
    double BeamCoincidenceHits(int index) const { return beamCoincidenceHits_[index];   }
    /// Returns the indexed hit time from muon veto scintillator M1
    double M1Hits(int index) const { return m1hits_[index];   }
    /// Returns the indexed hit time from muon veto scintillator M2
    double M2Hits(int index) const { return m2hits_[index];   }
    /// Returns the indexed hit time from muon veto scintillator M3
    double M3Hits(int index) const { return m3hits_[index];   }

    /// Returns the indexed hit time from scintillator S1, which is 12cm x 12cm.
    double S1Hits(int index) const { return s1hits_[index];   }
    /// Returns the indexed hit time from scintillator S2, which is 4cm x 4cm.
    double S2Hits(int index) const { return s2hits_[index];   }
    /// Returns the indexed hit time from scintillator S3, which is 2cm x 2cm.
    double S3Hits(int index) const { return s3hits_[index];   }
    /// Returns the indexed hit time from scintillator S4, which is 12cm x 12cm.
    double S4Hits(int index) const { return s4hits_[index];   }

    /// Returns the indexed hit time from beam halo counter UP HORIZONTAL.
    double BH1Hits(int index) const { return bh1hits_[index];   }
    /// Returns the indexed hit time from from beam halo counter BEAM LEFT FROM PARTICLE'S VIEW.
    double BH2Hits(int index) const { return bh2hits_[index];   }
    /// Returns the indexed hit time from beam halo counter BEAM RIGHT FROM PARTICLE'S VIEW.
    double BH3Hits(int index) const { return bh3hits_[index];   }
    /// Returns the indexed hit time from beam halo counter DOWN HORZINTAL.
    double BH4Hits(int index) const { return bh4hits_[index];   }
    /// Returns the hit in the specified channel 
    int V775(int index)  const {return V775_[index];}

    // Setter methods
    void   setTimes (const double trigger_time,
		     const double ttc_l1a_time,
		     const double laser_flash,
		     const double qie_phase,
		     const double TOF1S_time,
		     const double TOF1J_time,
		     const double TOF2S_time,
		     const double TOF2J_time);

    void   setHits  (const std::vector<double>& m1hits,
		     const std::vector<double>& m2hits,
		     const std::vector<double>& m3hits,
		     const std::vector<double>& s1hits,
		     const std::vector<double>& s2hits,
		     const std::vector<double>& s3hits,
		     const std::vector<double>& s4hits,
		     const std::vector<double>& bh1hits,
		     const std::vector<double>& bh2hits,
		     const std::vector<double>& bh3hits,
		     const std::vector<double>& bh4hits,
		     const std::vector<double>& beamCoincidenceHits);

  void setV775 (int *V775);


  private:
    double triggerTime_;
    double ttcL1Atime_;
    double laserFlash_;
    double qiePhase_;
    double TOF1Stime_;
    double TOF1Jtime_;
    double TOF2Stime_;
    double TOF2Jtime_;

    int V775_[32];
    std::vector<double> beamCoincidenceHits_;

    std::vector<double> m1hits_;
    std::vector<double> m2hits_;
    std::vector<double> m3hits_;

    std::vector<double> s1hits_;
    std::vector<double> s2hits_;
    std::vector<double> s3hits_;
    std::vector<double> s4hits_;

    std::vector<double> bh1hits_;
    std::vector<double> bh2hits_;
    std::vector<double> bh3hits_;
    std::vector<double> bh4hits_;
  };

  std::ostream& operator<<(std::ostream& s, const HcalTBTiming& htbtmg);

#endif
