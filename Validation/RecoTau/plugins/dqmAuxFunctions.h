#ifndef ElectroWeakAnalysis_EWKTau_dqmAuxFunctions_h
#define ElectroWeakAnalysis_EWKTau_dqmAuxFunctions_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include <string>
#include <vector>
#include <map>

const std::string parKeyword = "#PAR#";
const std::string plotKeyword = "#PLOT#";
const std::string rangeKeyword = "#RANGE";
const std::string processDirKeyword = "#PROCESSDIR#";

std::string replace_string(const std::string&, const std::string&, const std::string&, unsigned, unsigned, int&);

std::string format_vstring(const std::vector<std::string>& vs);

template <class T>
void readCfgParameter(const edm::ParameterSet& cfgParSet, std::map<std::string, T>& def)
{
  std::vector<std::string> cfgParNames = cfgParSet.getParameterNamesForType<edm::ParameterSet>();
  for ( std::vector<std::string>::const_iterator cfgParName = cfgParNames.begin(); 
	cfgParName != cfgParNames.end(); ++cfgParName ) {
    edm::ParameterSet cfgParDef = cfgParSet.getParameter<edm::ParameterSet>(*cfgParName);
    
    def.insert(std::pair<std::string, T>(*cfgParName, T(*cfgParName, cfgParDef)));
  }
}

std::string dqmDirectoryName(const std::string&);
std::string dqmSubDirectoryName_merged(const std::string&, const std::string&);
void dqmCopyRecursively(DQMStore&, const std::string&, const std::string&, double, int, bool);

const std::string dqmSeparator = "/";
//const std::string dqmRootDirectory = std::string(dqmSeparator).append("DQMData").append(dqmSeparator);
const std::string dqmRootDirectory = "";

void separateHistogramFromDirectoryName(const std::string&, std::string&, std::string&);

#endif
