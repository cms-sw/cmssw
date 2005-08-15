#ifndef UTILITIES_GENERAL_PROC_H
#define UTILITIES_GENERAL_PROC_H
//
#include <string>
#include <fstream>

namespace Capri {

struct Proc {

  typedef long long int d;
  typedef int i;
  typedef unsigned int u;


  static int instanceId();

  Proc(int pid=0);
  void refreshTime(d& ut, d&st);

  std::ifstream statf;
  std::string fname;

};

/** dump of /proc/self/stat
 */
struct ProcStat : public Proc {

  typedef Proc::d d;
  typedef int i;
  typedef unsigned int u;

  ProcStat();
  
  void refresh();

  i pid;
  std::string comm;
  char state;
  i ppid;
  i pgrp;
  i session;
  i tty;
  i tpgid;
  u flags;
  u minflt;
  u cminflt;
  u majflt;
  u cmajflt;
  d utime;
  d stime;
  d cutime;
  d cstime;
  d counter;

};

}

#endif // UTILITIES_GENERAL_PROC_H
