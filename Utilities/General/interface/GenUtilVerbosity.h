#ifndef GenUtilVerbosity_H
#define GenUtilVerbosity_H
//
//
//   V 0.0 
//

#include "Utilities/General/interface/envUtil.h"
#include<ostream>


/**  a trivial verbosity switch
 */
template<typename Tag>
class EnvVerbosity {
public:

  static bool on() { 
    return me();
  }

  static bool silent() { 
    return silence();
  }

  static void switchOn() { me()=true; silence()=false;}
  static void switchOff() { me()=false;}
  static void quiet() { silence()=true; me()=false;}

private:
  EnvVerbosity(){}

private:
  static envSwitch & me();
  static envSwitch & silence();
};

namespace GenUtil {

  struct Tag{};

  extern std::ostream cout;
}

typedef EnvVerbosity<GenUtil::Tag> GenUtilVerbosity ;

#endif // GenUtilVerbosity_H
