#include <iostream>
#include <cassert>
#include <cstdlib>

class ScheduleItems {
public:
  ScheduleItems() {}
  void initMisc();
};

void ScheduleItems::initMisc() { std::cout << "ScheduleItems::initMisc() called" << std::endl; }

void my_assert() { assert(false && "Intentional assert failure"); }

int main(int argc, char** argv) {
  if (argc > 1) {
    std::string_view opt{argv[1]};
    if (opt == "before") {
      my_assert();
    } else if (opt == "after") {
      ScheduleItems obj;
      obj.initMisc();
      my_assert();
    }
  }
  return 0;
}
