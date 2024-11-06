#include <string>
#include <FWCore/Utilities/interface/thread_safety_macros.h>

class Foo {
public:
  Foo() : m_intMutable(-1), m_strMutable("foo") {};
  int someMethod() const { return m_intMutable; }
  bool someOtherMethod() const { return m_strMutable.empty(); }

private:
  mutable int m_intMutable;
  mutable std::string m_strMutable;
  CMS_SA_ALLOW long m_longMutable;  // should not be reported
};

class Bar {
  void modifyingMethod(int j) const {
    m_intMutable = j;
    m_intMutable++;
    m_intMutable *= j;
    m_intMutable--;
    if (j != 0) {
      m_intMutable /= j;
    }
  }
  void otherModifyingMethod(std::string& other) const { m_strMutable = other; }

private:
  mutable int m_intMutable;
  mutable std::string m_strMutable;
};

int main() { return 0; }
