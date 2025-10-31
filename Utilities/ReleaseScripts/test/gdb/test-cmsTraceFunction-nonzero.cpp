#include <iostream>
#include <cstdlib>

class ScheduleItems {
public:
  ScheduleItems() {}
  void initMisc();
};

void ScheduleItems::initMisc() { std::cout << "ScheduleItems::initMisc() called" << std::endl; }

int main(int argc, char** argv) {
  if (argc > 1) {
    std::string_view opt{argv[1]};
    if (opt == "before") {
      return 1;
    } else if (opt == "after") {
      ScheduleItems obj;
      obj.initMisc();
      return 1;
    }
  }
  return 0;
}
