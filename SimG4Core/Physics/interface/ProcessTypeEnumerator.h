#ifndef ProcessTypeEnumerator_H
#define ProcessTypeEnumerator_H

#include<map>
#include<string>
/**
 * This is the Profound level of ProcessTypeEnumerator. It maps between int and 
 * our (CMS) convention of physical processes.
 */
class ProcessTypeEnumerator {
 public:
  typedef std::map<std::string,unsigned int> MapType;  
  typedef std::map<unsigned int,std::string> ReverseMapType;

  ProcessTypeEnumerator();
  unsigned int processId(std::string);
  std::string processName(unsigned int);
  unsigned int numberOfKnownProcesses();
 private:
  void buildReverseMap();
  MapType mapProcesses;
  ReverseMapType reverseMapProcesses;
};

#endif 

