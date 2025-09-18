#include <iostream>
#include <cassert>
#include <cstdlib>
#include <csignal>

class ScheduleItems {
public:
  ScheduleItems() {}
  void initMisc();
};

void ScheduleItems::initMisc() { std::cout << "ScheduleItems::initMisc() called" << std::endl; }

void my_segfault() { raise(SIGSEGV); }

int main(int argc, char** argv) {
  if (argc > 1) {
    std::string_view opt{argv[1]};
    if (opt == "before") {
      my_segfault();
    } else if (opt == "after") {
      ScheduleItems obj;
      obj.initMisc();
      my_segfault();
    }
  }
  return 0;
}
