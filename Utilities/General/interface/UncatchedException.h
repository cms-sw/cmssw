#ifndef UncatchedException_H
#define UncatchedException_H
//
//
//   V 0.0 
//

#include "Utilities/General/interface/MutexUtils.h"
#include <iosfwd>
namespace cms {
   class Exception;
}

/**
 */
class UncatchedException {
public:

  UncatchedException();
  explicit UncatchedException(const cms::Exception & err);
  static void dump(std::ostream & o, bool det=false);
  static void rethrow();
  static int count();

private:

  static cms::Exception * it;

  static LockMutex::Mutex mutex;
  static int count_; 

};



#endif // UncatchedException_H
