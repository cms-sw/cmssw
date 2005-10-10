#ifndef CONFIGURATIONRECORD_H
#define CONFIGURATIONRECORD_H
//
// a parsed configuration record
//
//    V 0.2   27/01/00
//        evaluation of env var added



#include <string>



#include <vector>
#include <utility>
 
/// forward  referece
#include<iosfwd>

/// a Configuration Record
/** parse a Record of the type
KEY separator FIELD
and store KEY&FIELD pairs in a vector
 */
class ConfigurationRecord {

public:
  class Keys {
  public:
    typedef std::vector<std::string>::const_iterator iter;
  public:
    Keys();
    explicit Keys(const std::string & isep);
    Keys(const std::string & isep, const std::vector<std::string> & imod, const std::vector<std::string> & icomm);

    const std::string & separator() const { return separator_;}
    const std::vector<std::string> & modifiers() const { return modifiers_;}
    const std::vector<std::string> & comments() const { return comments_;}

  private:
    std::string separator_;
    std::vector<std::string> modifiers_;
    std::vector<std::string> comments_;

  };

  static const Keys defKeys;

public:
  typedef std::vector<std::pair<std::string,std::string> > Dict; 
  typedef Dict::const_iterator DictCI;
  typedef Dict::const_reverse_iterator DictCRI;
  typedef Dict::iterator DictI;
  typedef Dict::value_type DictEl;

public:

  explicit ConfigurationRecord(const Keys & ik=defKeys, bool evEnVar=true);

  ConfigurationRecord(std::istream & input, const Keys & ik=defKeys, bool evEnVar=true);

  ConfigurationRecord(const std::string & isource, const Keys & ik=defKeys, bool evEnVar=true);

  DictCI begin() const { return dict_.begin();}
  DictCI end()   const { return dict_.end();}

  DictCRI rbegin() const { return dict_.rbegin();}
  DictCRI rend()   const { return dict_.rend();}

  void dump() const;

protected:

  void add(const std::string & isource);
  void add(std::istream & input);

  void parse();


protected:
  Dict dict_;

private:
  std::string source_;
  Keys keys;
  bool   evalEnVar_;

};



#endif  // CONFIGURATIONRECORD_H
