#ifndef UncatchedException_H
#define UncatchedException_H
//
//
//   V 0.0 
//

#include "Utilities/General/interface/MutexUtils.h"
#include <iosfwd>
namespace seal {
  class Error;
}

/**
 */
class UncatchedException {
public:

  UncatchedException();
  explicit UncatchedException(const seal::Error & err);
  static void dump(std::ostream & o, bool det=false);
  static void rethrow();
  static int count();

private:

  static seal::Error * it;

  static LockMutex::Mutex mutex;
  static int count_; 

};



#endif // UncatchedException_H
