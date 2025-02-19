#ifndef UTILITIES_GENERAL_ENVUTIL_H
#define UTILITIES_GENERAL_ENVUTIL_H
//
//   small utility to handle
//   enviromental variables
//
//   Version 0.0      V.I.  5/5/98
//
//   Version 1.0      V.I.  11/11/01
//     also set env
//     much less inlined
//

#include <string>
#include <iosfwd>

/** a class to handle enviromental variables
 */
class envUtil {

public:

  /// default constructor
  envUtil(){}

  /** constructor by the evironmental variable name and its
      default value (used if the environmental variable is not set)
   */
  envUtil(const char * envName, const char * defaultValue="");

  /** constructor by the evironmental variable name and its
      default value (used if the environmental variable is not set)
   */
  envUtil(const char * envName, const std::string & defaultValue);
 
  /// constructor from the default value only
  envUtil(const std::string & defaultValue) :  env_(defaultValue) {}


  /// assignement operator (change the value...)
  envUtil & operator = (const std::string& s) { env_ = s; return *this;}

  // operator const string & () const { return _env;}

  /// cast in char * (for old interface and c-like interface)
  operator const char * () const { return env_.c_str();}

  /// return the value 
  const std::string &  getEnv() const { return env_;}

  // set env to current value;
  void setEnv();

  // set env to arg
  void setEnv(const std::string & nval);

  /// acts as the constructor with the same signature
  const std::string &  getEnv(const char * envName, const char * defaultValue="");
  
  const std::string & name() const { return envName_;}

protected:

  std::string envName_;
  std::string env_;

};

/** a bool controlled by envvar
 */
class envSwitch {
public:

  /// constructor from env var name
  envSwitch(const char * envName);

  /// assignement operator (change the value...)
  envSwitch & operator = (bool b) { it=b; return *this;}
  

  /// return the value 
  operator const bool& () const { return it;}

  /// return the value 
  const bool &  getEnv() const { return it;}

  const std::string & name() const { return envName_;}

private:

  std::string envName_;
  bool it;

};

std::ostream & operator<<(std::ostream & co, const envUtil & eu);
std::istream & operator>>(std::istream & ci, envUtil & eu);

#endif // UTILITIES_GENERAL_ENVUTIL_H
