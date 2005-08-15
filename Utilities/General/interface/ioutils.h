#ifndef UTILITIES_GENERAL_IOUTILS_H
#define UTILITIES_GENERAL_IOUTILS_H
//
// general io utilities...
// no
//  V 1.99999 o  18/04/02
//  one more change to oss for linux...
//  V 1.999999 o  09/08/02
//  one more change to oss for linux...
//
#include <cstdio>
#include <unistd.h>
#include <cstdlib>

#include <sstream>
#include <iterator>
#include <string>
#include <vector>

// generic ato
template<class T>
struct ato {
  inline const T & operator () (const T&c) const { return c;}
};

template<>
struct ato<bool> {
  bool operator () (const std::string &c) const;
};

template<>
struct ato<int> {
  inline int operator () (const char * c) const { return atoi(c);}
  inline int operator () (const std::string &c) const { return atoi(c.c_str());}
};

template<>
struct ato<unsigned int> {
  inline unsigned int operator () (const char * c) const { return strtoul(c,0,0);}
  inline unsigned int operator () (const std::string &c) const { return strtoul(c.c_str(),0,0);}
};

template<>
struct ato<float> {
  inline float operator () (const char * c) const { return atof(c);}
  inline float operator () (const std::string &c) const { return atof(c.c_str());}
};

template<>
struct ato<double> {
  inline double operator () (const char * c) const { return strtod(c,0);}
  inline double operator () (const std::string &c) const { return strtod(c.c_str(),0);}
};

template<>
struct ato<char *> {
  inline const char * operator () (const char * c) const { return c;}
  inline const char * operator () (const std::string &c) const { return c.c_str();}
};

template<>
struct ato<std::string> {
  inline const std::string & operator () (const char * c) const { static std::string cs; cs=c;return cs;}
  inline const std::string & operator () (const std::string &c) const { return c;}
};

template<typename T>
struct ato<std::vector<T> > {
  inline std::vector<T> operator () (const char * c) const { 
    std::vector<T> v;
    if (!c) return v;
    std::istringstream in(c);
    std::istream_iterator<T> sbegin(in), send;
    std::copy(sbegin,send,std::inserter(v,v.end()));
    return v;
  }
  
  inline std::vector<T>  operator () (const std::string &cs) const { 
    std::vector<T> v;
    if (cs.empty()) return v;
    std::istringstream in(cs.c_str());
    std::istream_iterator<T> sbegin(in), send;
    std::copy(sbegin,send,std::inserter(v,v.end()));
    return v;
  }
 
};


/** a simple class to write ints, floats and doubles in chars. 

 Usage as in:  string mystring; ... ; mystring += toa()(4.);   
 */
class toa {

public:

  inline toa(){}
  ~toa();


  const char * operator () (const unsigned short int&c) const { return operator()(int(c));}
  const char * operator () (const short int&c) const { return operator()(int(c));}
  const char * operator () (const unsigned long&c) const { return operator()(int(c));}
  const char * operator () (const long&c) const { return operator()(int(c));}
  const char * operator () (const unsigned int&c) const { return operator()(int(c));}
  const char * operator () (bool c) const;
  const char * operator () (const int&c) const;
  const char * operator () (const float&c) const { return operator()(double(c));}
  const char * operator () (const double&c) const;
  const char * operator () (const char c[], int n) const;
  const char * operator () (const char *c) const { return c;}
  const char * operator () (const std::string&c) const { return c.c_str();}
  template<typename T>
  const char * operator () (const std::vector<T>&v) const { 
    std::ostream_iterator<T > oi(oss()," ");
    std::copy(v.begin(),v.end(),oi);
    localS() = oss().str();
    return localS().c_str();
  }

private:
  mutable std::ostringstream oss_;
  std::ostringstream & oss() const { return oss_;}

  static std::string & localS();

private:
  toa(const toa&){}
  void operator=(const toa&){}
};

#endif // UTILITIES_GENERAL_IOUTILS_H
