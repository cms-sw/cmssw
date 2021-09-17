#ifndef HCALTBRUNDATA_H
#define HCALTBRUNDATA_H 1

#include <string>
#include <iostream>

/** \class HcalTBRunData

  This class contains data associated with a run, such as character
  strings describing the run type, beam mode, and also the beam energy.

  $Date: 2005/10/06 22:21:33 $
  $Revision: 1.2 $
  \author P. Dudero - Minnesota
  */
class HcalTBRunData {
public:
  HcalTBRunData();

  // Getter methods
  /// Returns the run type string
  const std::string& runType() const { return runType_; }
  /// Returns the beam mode string
  const std::string& beamMode() const { return beamMode_; }

  /// Returns the beam energy in GeV
  double beamEnergyGeV() const { return beamEnergyGeV_; }

  // Setter methods
  void setRunData(const char* run_type, const char* beam_mode, double beam_energy_gev);

private:
  std::string runType_;
  std::string beamMode_;
  double beamEnergyGeV_;
};

std::ostream& operator<<(std::ostream& s, const HcalTBRunData& htbrd);

#endif
