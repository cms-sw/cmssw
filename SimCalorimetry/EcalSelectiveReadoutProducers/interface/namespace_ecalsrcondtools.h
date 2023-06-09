#ifndef namespace_ecalsrcondtools_h
#define namespace_ecalsrcondtools_h

/*
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/EcalObjects/interface/EcalSRSettings.h"

#include <string>
#include <iostream>
/**
 */
namespace ecalsrcondtools {
  /** Converts CMSSW python file selective readout setting ("parameter set")
   * into a condition database object.
   * Configuration from parameter set covers only part of the config,
   * mainly the configuration needed for SR emulation in the MC. The parameters
   * not supported by python configuration are left untouched in the sr object.
   * @param sr [in] ECAL selective readout setting object to set
   * @param ps CMSSW parameter set containing the SR parameters to set
   */
  void importParameterSet(EcalSRSettings& sr, const edm::ParameterSet& ps);

  /** Imports an SRP configuration file (stored in database "CLOB") into a
   * Selective readout setting object.
   * @param sr [in] ECAL selective readout setting object to set
   * @param f configuration file stream. A stringstream can be used if the configuration
   * is available as an stl string of a c-string: stringstream buf; buf << s;
   * @param debug verbosity flag. If true, imported parameter are displayed on stdout.
   */
  void importSrpConfigFile(EcalSRSettings& sr, std::istream& f, bool debug = false);

  ///convert hardware weights (interger weights)
  ///into normalized weights. The former reprensentation is used in DCC firmware
  ///and in online databaser, while the later is used in offline software.
  double normalizeWeights(int hwWeight);
};  // namespace ecalsrcondtools

#endif  //SRCONDACCESS_H not defined
