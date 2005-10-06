#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"

using namespace std;

  HcalTBTiming::HcalTBTiming() :
    triggerTime_(0),
    ttcL1Atime_(0),
    beamCoincidence_(0),
    laserFlash_(0),
    qiePhase_(0),
    m1hits_(),
    m2hits_(),
    m3hits_(),
    s1hits_(),
    s2hits_(),
    s3hits_(),
    s4hits_() {
  }

  void HcalTBTiming::setTimes (const double trigger_time,
			       const double ttc_l1a_time,
			       const double beam_coincidence,
			       const double laser_flash,
			       const double qie_phase) {
    triggerTime_     = trigger_time;
    ttcL1Atime_      = ttc_l1a_time;
    beamCoincidence_ = beam_coincidence;
    laserFlash_      = laser_flash;
    qiePhase_        = qie_phase;
  }

  void HcalTBTiming::setHits  (const std::vector<double>& m1hits,
			       const std::vector<double>& m2hits,
			       const std::vector<double>& m3hits,
			       const std::vector<double>& s1hits,
			       const std::vector<double>& s2hits,
			       const std::vector<double>& s3hits,
			       const std::vector<double>& s4hits) {
    m1hits_ = m1hits;
    m2hits_ = m2hits;
    m3hits_ = m3hits;

    s1hits_ = s1hits;
    s2hits_ = s2hits;
    s3hits_ = s3hits;
    s4hits_ = s4hits;
  }

  ostream& operator<<(ostream& s, const HcalTBTiming& htbtmg) {

    s << "Trigger time     = " << htbtmg.triggerTime() << endl;
    s << "TTC L1A time     = " << htbtmg.ttcL1Atime() << endl;
    s << "Beam Coincidence = " << htbtmg.beamCoincidence() << endl;
    s << "Laser Flash      = " << htbtmg.laserFlash() << endl;
    s << "QIE Phase        = " << htbtmg.qiePhase() << endl;

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

    return s;
  }
