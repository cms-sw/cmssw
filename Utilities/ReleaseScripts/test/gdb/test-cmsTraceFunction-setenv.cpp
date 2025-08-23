#include <iostream>
#include <cstdlib>
#include <stdlib.h>

class ScheduleItems {
public:
  ScheduleItems() {}
  void initMisc();
};

void ScheduleItems::initMisc() { std::cout << "ScheduleItems::initMisc() called" << std::endl; }

void my_setenv(const char* env, const char* value) {
  setenv(env, value, 1);
  std::cout << "setenv() called" << std::endl;
  std::cout << env << "=" << std::getenv(env) << std::endl;
}

int main() {
  my_setenv("FOO", "1");
  ScheduleItems obj;
  obj.initMisc();
  my_setenv("FOO", "2");
  my_setenv("FOO", "3");
  return 0;
}
