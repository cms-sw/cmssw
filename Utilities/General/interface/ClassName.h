#ifndef UTILITIES_GENERAL_CLASSNAME_H
#define UTILITIES_GENERAL_CLASSNAME_H
//
#include <typeinfo>
#include <string>
#include <cstdlib>

inline std::string firstNonNumeric(const char * sc) {
  std::string s(sc);
  size_t  pos = s.find_first_not_of("0123456789");
  s.erase(0,pos);
  return s;
}

class Demangle {
public:
  Demangle(const char * sc);

  const char * operator()() const { return demangle;}
  
  ~Demangle() { if (demangle) free((void*)demangle); }
  
private:
  
  char * demangle;
};


template<class T>
inline  std::string className(const T&t) { 
  return std::string(Demangle(typeid(t).name())());
}

template<class T>
class ClassName {
public:
  static const std::string & name() {
    static const std::string ln(Demangle(typeid(T).name())());
    return ln;
  }
};

#endif // UTILITIES_GENERAL_CLASSNAME_H
