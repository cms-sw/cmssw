#include "FWCore/Utilities/interface/thread_safety_macros.h"

class Bar {
public:
  void someMethod(int val) { value = val; }

private:
  int value;
};

// will produce warnings by MutableMemberChecker
class Foo {
public:
  void nonConstInConst(int val) const { barMutableMember.someMethod(val); }

  // Shall not produce warning
  CMS_THREAD_SAFE void nonConstInConst_safe(int val) const { barMutableMember.someMethod(val); }

  void changeMutableInConst1(double val) const { privateMutable = val; }
  void changeMutableInConst2(double val) const { privateMutable += val; }
  void changeMutableInConst3(double val) const { privateMutable *= val; }
  void changeMutableInConst4(double val) const { privateMutable /= val; }
  void changeMutableInConst5(double val) const { privateMutable -= val; }
  void changeMutableInConst6(double val) const { privateMutable++; }
  void changeMutableInConst7(double val) const { privateMutable--; }

  CMS_THREAD_SAFE void changeMutableInConst_safe(double val) { privateMutable = val * val; };

  mutable int badPublicMutable;

private:
  mutable Bar barMutableMember;
  mutable double privateMutable;
  int goodMember;
};

int main() { return 0; }
