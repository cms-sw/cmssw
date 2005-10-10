#include<iostream>
#include "Utilities/General/interface/pidInfo.h"

const unsigned int pidInfo::pagesz=4;

#include "Utilities/Threads/interface/ThreadUtils.h"

pidInfo::pidInfo(const std::string & s) {
  std::ostream & co = TSafeOstream();
  co <<"Memory use in kb," <<s<< std::endl;
  load(&co);
}

pidInfo::pidInfo(std::ostream & co, const std::string & s) {
  co <<"Memory use in kb," <<s<< std::endl;
  load(&co);

}

pidInfo::pidInfo() {
  load(0);
}

#include <fstream>
void pidInfo::load(std::ostream * co) {
  unsigned int sharedsz;
  unsigned int textsz;
  unsigned int datasz;
  unsigned int libsz;
  unsigned int dtsz;

  std::ifstream statm("/proc/self/statm");
  statm >> memsz >> rsssz >> sharedsz >> textsz >> datasz
	>> libsz >> dtsz;
  statm.close();

  if (!co) return;


  (*co) << "Total memory=" << pagesz*memsz
	<< ", Resident=" << pagesz*rsssz
	<< ", Shared=" << pagesz*sharedsz
	<< ", Text=" << pagesz*textsz
	<< ", Data=" << pagesz*datasz
	<< ", Libraries=" << pagesz*libsz
	<< ", Dirty pages=" << pagesz*dtsz
	<< std::endl;
}

