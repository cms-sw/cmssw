#include <iostream>
#include <cstdlib>
#include <cstring>
#include <stdlib.h>

class ScheduleItems {
public:
  ScheduleItems() {}
  void initMisc();
};

void ScheduleItems::initMisc() { std::cout << "ScheduleItems::initMisc() called" << std::endl; }

void my_putenv(const char* env, char* put) {
  putenv(put);
  std::cout << "putenv() called" << std::endl;
  std::cout << env << "=" << std::getenv(env) << std::endl;
}

int main() {
  // putenv() expects modifiable char array that lives throughout the program
  char* foo1 = new char[10];
  char* foo2 = new char[10];
  char* foo3 = new char[10];
  std::strncpy(foo1, "FOO=1", 10);
  std::strncpy(foo2, "FOO=2", 10);
  std::strncpy(foo3, "FOO=3", 10);

  my_putenv("FOO", foo1);
  ScheduleItems obj;
  obj.initMisc();
  my_putenv("FOO", foo2);
  my_putenv("FOO", foo3);
  return 0;
}
