#include "TBDataFormats/HcalTBObjects/interface/HcalTBRunData.h"

using namespace std;

  HcalTBRunData::HcalTBRunData() :
    runType_(""),
    beamMode_(""),
    beamEnergyGeV_(0.0) {
  }

  void HcalTBRunData::setRunData( const char *run_type,
				  const char *beam_mode,
				  double      beam_energy_gev ) {
    runType_       = run_type;
    beamMode_      = beam_mode;
    beamEnergyGeV_ = beam_energy_gev;
  }

  ostream& operator<<(ostream& s, const HcalTBRunData& htbrd) {
    s << "Run type    = " << htbrd.runType()  << endl;
    s << "Beam Mode   = " << htbrd.beamMode() << endl;
    s << "Beam Energy = " << htbrd.beamEnergyGeV() << " GeV" << endl;
    return s;
  }
