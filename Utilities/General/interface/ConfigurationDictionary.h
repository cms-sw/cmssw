#ifndef CONFIGURATIONDictionary_H
#define CONFIGURATIONDictionary_H
//
// a Dictionary of Configuration strings
//
//

#include <map>
#include <string>
#include <iosfwd>

class ConfigurationRecord;

class ConfigurationDictionary : public std::map<std::string, std::string >{
public:
  typedef std::map<std::string, std::string > super;
  typedef super::const_iterator const_iterator;
  typedef std::pair<const_iterator, const_iterator> CRange;
public:
  ConfigurationDictionary(){}
  ConfigurationDictionary(const ConfigurationRecord& conf);
  void add (const ConfigurationRecord& conf);
  void dump() const;
  void dump(std::ostream & co) const;
  /// return the range of all items starting with suffix+sep
  CRange partial_range(const std::string & prefix, char sep=':') const;
};

#endif // CONFIGURATIONDictionary_H
