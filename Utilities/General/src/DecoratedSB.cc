#include "Utilities/General/interface/DecoratedSB.h"


BaseDecoratedSB::BaseDecoratedSB(std::streambuf * isb) : me(0), sb_(isb) {
}
BaseDecoratedSB::~BaseDecoratedSB(){}
  
#include "Utilities/Threads/interface/ThreadUtils.h"
  
int BaseDecoratedSB::sync() {
  static LockMutex::Mutex mutex;
  int jj = 
    std::stringbuf::sync();
  {
    LockMutex gl(mutex);
    pre(me);
    me.rdbuf(sb_);
    pre(me);
    me << (*this).str();
    (*this).str("");
    post(me);
    me.flush();
    me.rdbuf(0);
  }
  return jj;
}
