#include "Utilities/General/interface/ConfigurationRecord.h"
#include "Utilities/General/interface/ConfigurationDictionary.h"

#include "Utilities/General/interface/GenUtilVerbosity.h"

using std::string;

ConfigurationDictionary::ConfigurationDictionary(const ConfigurationRecord& conf) {
  add(conf);
}

void ConfigurationDictionary::add(const ConfigurationRecord& conf) {
  for (ConfigurationRecord::DictCI p=conf.begin(); p!=conf.end();++p) {
    if ( (*p).first[(*p).first.size()-1]=='+' ) 
      (*this)[(*p).first.substr(0,(*p).first.size()-1)] += string(" ") + (*p).second;
    else (*this)[(*p).first] = (*p).second;
  }
}

void ConfigurationDictionary::dump() const {
  dump(GenUtil::cout);
}

void ConfigurationDictionary::dump(std::ostream& co) const {
  for (const_iterator p=begin(); p!=end(); ++p)
    co << (*p).first << " = " << (*p).second << "\n";
  co.flush();
} 


ConfigurationDictionary::CRange ConfigurationDictionary::partial_range(const string & prefix, char sep) const {
  
  string l1 = prefix + sep;
  sep++;
  string l2 = prefix + sep;
  return CRange(lower_bound(l1),lower_bound(l2));
  
}
