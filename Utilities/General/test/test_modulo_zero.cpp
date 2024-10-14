#include <csignal>
#include <iostream>
#include <stdexcept>

void signalHandler(int signal) {
  if (signal == SIGFPE) {
    std::cerr << "Floating point exception caught!" << std::endl;
    exit(0);
  }
}

int main() {
  int binLabelCounter = 100;
  std::signal(SIGFPE, signalHandler);
  for (int nBookedBins = 1; nBookedBins < 1000000; nBookedBins++) {
    int res = binLabelCounter % ((int)(nBookedBins / 25));
    std::cout << nBookedBins << "% 0 = " << res << std::endl;
  }
  return 1;
}
