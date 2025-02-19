#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"

using namespace std;

  HcalTBTiming::HcalTBTiming() :
    triggerTime_(0),
    ttcL1Atime_(0),
    laserFlash_(0),
    qiePhase_(0),
    TOF1Stime_(0),
    TOF1Jtime_(0),
    TOF2Stime_(0),
    TOF2Jtime_(0),
    beamCoincidenceHits_(),
    m1hits_(),
    m2hits_(),
    m3hits_(),
    s1hits_(),
    s2hits_(),
    s3hits_(),
    s4hits_(),
    bh1hits_(),
    bh2hits_(),
    bh3hits_(),
    bh4hits_() {
    for (int i=0;i<32;i++)V775_[i]=-1;
  }

  void HcalTBTiming::setTimes (const double trigger_time,
			       const double ttc_l1a_time,
			       const double laser_flash,
			       const double qie_phase,
                               const double TOF1S_time,
			       const double TOF1J_time,
			       const double TOF2S_time,
			       const double TOF2J_time) {
    triggerTime_     = trigger_time;
    ttcL1Atime_      = ttc_l1a_time;
    laserFlash_      = laser_flash;
    qiePhase_        = qie_phase;
    TOF1Stime_       = TOF1S_time;
    TOF1Jtime_       = TOF1J_time;
    TOF2Stime_       = TOF2S_time;
    TOF2Jtime_       = TOF2J_time;
  }

  void HcalTBTiming::setHits  (const std::vector<double>& m1hits,
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
			       const std::vector<double>& beamCoincidenceHits) {
    m1hits_ = m1hits;
    m2hits_ = m2hits;
    m3hits_ = m3hits;

    s1hits_ = s1hits;
    s2hits_ = s2hits;
    s3hits_ = s3hits;
    s4hits_ = s4hits;

    bh1hits_ = bh1hits;
    bh2hits_ = bh2hits;
    bh3hits_ = bh3hits;
    bh4hits_ = bh4hits;
    beamCoincidenceHits_ = beamCoincidenceHits;
  }

    void HcalTBTiming::setV775 (int *V775)
 {
    for (int i=0;i<32;i++)V775_[i]=V775[i];
 }

  
  ostream& operator<<(ostream& s, const HcalTBTiming& htbtmg) {

    s << "Trigger time     = " << htbtmg.triggerTime() << endl;
    s << "TTC L1A time     = " << htbtmg.ttcL1Atime() << endl;
    s << "Laser Flash      = " << htbtmg.laserFlash() << endl;
    s << "QIE Phase        = " << htbtmg.qiePhase() << endl;
    s << "TOF1S            = " << htbtmg.TOF1Stime() << endl;
    s << "TOF1J            = " << htbtmg.TOF1Jtime() << endl;
    s << "TOF2S            = " << htbtmg.TOF2Stime() << endl;
    s << "TOF2J            = " << htbtmg.TOF2Jtime() << endl;

    s << "M1 hits: ";
    for (int i=0; i<htbtmg.M1Count(); i++) {
      if (i) s << ", ";
      s << htbtmg.M1Hits(i);
    }
    s << endl;

    s << "M2 hits: ";
    for (int i=0; i<htbtmg.M2Count(); i++) {
      if (i) s << ", ";
      s << htbtmg.M2Hits(i);
    }
    s << endl;

    s << "M3 hits: ";
    for (int i=0; i<htbtmg.M3Count(); i++) {
      if (i) s << ", ";
      s << htbtmg.M3Hits(i);
    }
    s << endl;

    s << "S1 hits: ";
    for (int i=0; i<htbtmg.S1Count(); i++) {
      if (i) s << ", ";
      s << htbtmg.S1Hits(i);
    }
    s << endl;

    s << "S2 hits: ";
    for (int i=0; i<htbtmg.S2Count(); i++) {
      if (i) s << ", ";
      s << htbtmg.S2Hits(i);
    }
    s << endl;

    s << "S3 hits: ";
    for (int i=0; i<htbtmg.S3Count(); i++) {
      if (i) s << ", ";
      s << htbtmg.S3Hits(i);
    }
    s << endl;

    s << "S4 hits: ";
    for (int i=0; i<htbtmg.S4Count(); i++) {
      if (i) s << ", ";
      s << htbtmg.S4Hits(i);
    }
    s << endl;

    s << "BH1 hits: ";
    for (int i=0; i<htbtmg.BH1Count(); i++) {
      if (i) s << ", ";
      s << htbtmg.BH1Hits(i);
    }
    s << endl;

    s << "BH2 hits: ";
    for (int i=0; i<htbtmg.BH2Count(); i++) {
      if (i) s << ", ";
      s << htbtmg.BH2Hits(i);
    }
    s << endl;

    s << "BH3 hits: ";
    for (int i=0; i<htbtmg.BH3Count(); i++) {
      if (i) s << ", ";
      s << htbtmg.BH3Hits(i);
    }
    s << endl;

    s << "BH4 hits: ";
    for (int i=0; i<htbtmg.BH4Count(); i++) {
      if (i) s << ", ";
      s << htbtmg.BH4Hits(i);
    }
    s << endl;

    s << "Beam Coincidence hits: ";
    for (int i=0; i<htbtmg.BeamCoincidenceCount(); i++) {
      if (i) s << ", ";
      s << htbtmg.BeamCoincidenceHits(i);
    }
    s << endl;

    return s;
  }
