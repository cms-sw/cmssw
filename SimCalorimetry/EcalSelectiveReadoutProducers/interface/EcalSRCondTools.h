#ifndef SRCONDACCESS_H
#define SRCONDACCESS_H

/*
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondFormats/EcalObjects/interface/EcalSRSettings.h"
#include "CondFormats/DataRecord/interface/EcalSRSettingsRcd.h"

/**
 */
class EcalSRCondTools : public edm::one::EDAnalyzer<> {
  //methods
public:
  /** Constructor
   * @param ps analyser configuration
   */
  EcalSRCondTools(const edm::ParameterSet&);

  /** Destructor
   */
  ~EcalSRCondTools() override;

  /** Called by CMSSW event loop
   * @param evt the event
   * @param es events setup
   */
  void analyze(const edm::Event& evt, const edm::EventSetup& es) override;

  /** Converts CMSSW python file selective readout setting ("parameter set")
   * into a condition database object.
   * Configuration from parameter set covers only part of the config,
   * mainly the configuration needed for SR emulation in the MC. The parameters
   * not supported by python configuration are left untouched in the sr object.
   * @param sr [in] ECAL selective readout setting object to set
   * @param ps CMSSW parameter set containing the SR parameters to set
   */
  static void importParameterSet(EcalSRSettings& sr, const edm::ParameterSet& ps);

  /** Imports an SRP configuration file (stored in database "CLOB") into a
   * Selective readout setting object.
   * @param sr [in] ECAL selective readout setting object to set
   * @param f configuration file stream. A stringstream can be used if the configuration
   * is available as an stl string of a c-string: stringstream buf; buf << s;
   * @param debug verbosity flag. If true, imported parameter are displayed on stdout.
   */
  static void importSrpConfigFile(EcalSRSettings& sr, std::istream& f, bool debug = false);

  ///convert hardware weights (interger weights)
  ///into normalized weights. The former reprensentation is used in DCC firmware
  ///and in online databaser, while the later is used in offline software.
  static double normalizeWeights(int hwWeight);

private:
  ///Help function to tokenize a string
  ///@param s string to parse
  ///@delim token delimiters
  ///@pos internal string position pointer. Must be set to zero before the first call
  static std::string tokenize(const std::string& s, const std::string& delim, int& pos);

  ///Help function to trim spaces at beginning and end of a string
  ///@param s string to trim
  static std::string trim(std::string s);

  //fields
private:
  edm::ParameterSet ps_;
  edm::ESGetToken<EcalSRSettings, EcalSRSettingsRcd> hSrToken_;

  bool done_;
};

#endif  //SRCONDACCESS_H not defined
