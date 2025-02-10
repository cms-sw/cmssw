#include "test-clang-tidy.h"
class BaseClass : public BaseClass2 {
public:
  BaseClass(int x) {
    m_x = x;
    ch = nullptr;
  };
  void override_func() override {}
  ~BaseClass() override;
  virtual int someMethod();

protected:
  int m_x;
  char* ch;
};

BaseClass::~BaseClass() {
  if (ch != nullptr) {
    delete ch;
    ch = nullptr;
  }
}
int BaseClass::someMethod() { return m_x; }

class DrivedClass : public BaseClass {
public:
  DrivedClass(int x) : BaseClass(x) {};
  ~DrivedClass() override;
  int someMethod() override;
};

DrivedClass::~DrivedClass() {}
int DrivedClass::someMethod() { return m_x * 2; }
