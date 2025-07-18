#include "test-clang-tidy.h"
class BaseClass : public BaseClass2 {
public:
  BaseClass(int x) {
    m_x = x;
    ch = 0;
  };
  void override_func() {}
  virtual ~BaseClass();
  virtual int someMethod();

protected:
  int m_x;
  char* ch;
};

BaseClass::~BaseClass() {
  if (ch != 0) {
    delete ch;
    ch = 0;
  }
}
int BaseClass::someMethod() { return m_x; }

class DrivedClass : public BaseClass {
public:
  DrivedClass(int x) : BaseClass(x) {};
  virtual ~DrivedClass();
  virtual int someMethod();
};

DrivedClass::~DrivedClass() {}
int DrivedClass::someMethod() { return m_x * 2; }
