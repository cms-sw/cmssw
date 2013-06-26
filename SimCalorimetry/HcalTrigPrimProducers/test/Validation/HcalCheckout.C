#include <cstdlib>
#include <string>
#include "HcalMonitor.h"
#include "HcalCompare.h"

using namespace std;
int main(int argc, char *argv[])
{
  cout << "argv[0] = " << argv[0] << " argv[1] = " << argv[1] << " argv[2] = " << argv[2] << " argv[3] " << argv[3] << "\n";
  string option = argv[1];
  if (option == "-m")
      {
	cout << "running monitor\n";
	monitor(argv[2],argv[3]);
      }
  else if (option == "-c")
    {
      cout << "running compare\n";
      compare(argv[2],argv[3]);
    }
  else
    {
      exit(1);
    }
  return 0;
}
