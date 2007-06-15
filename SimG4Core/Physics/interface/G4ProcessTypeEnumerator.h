#ifndef G4ProcessTypeEnumerator_H
#define G4ProcessTypeEnumerator_H

#include<map>
#include<vector>
#include<string>

class ProcessTypeEnumerator;
class G4VProcess;
/**
 * This is the Mantis level of ProcessTypeEnumerator. It maps betweenG4 and 
 * our (CMS) convention of physical processes.
 */
class G4ProcessTypeEnumerator {

public:
  //
  // MapType: G4->CMS (1 to 1)
  // ReverseMapType: CMS->G4 (1 to many)
  //
  typedef std::map<std::string,std::string> MapType;  
  typedef std::map<std::string,std::vector<std::string> > ReverseMapType;
  /**
   * This is the only method to be called by the sensitive detector
   */
  unsigned int processId(const G4VProcess*);
  int processIdLong(const G4VProcess*);

  G4ProcessTypeEnumerator();
  ~G4ProcessTypeEnumerator();
  std::string processCMSName(std::string);
  std::vector<std::string> processG4Name(std::string);
  std::string processG4Name(int);
  unsigned int numberOfKnownG4Processes();
  unsigned int numberOfKnownCMSProcesses();

private:
  ProcessTypeEnumerator* theProcessTypeEnumerator;
  void buildReverseMap();
  MapType mapProcesses;
  ReverseMapType reverseMapProcesses;
  std::map<std::string,int> map2Process;
  std::map<int,std::string> reverseMap2Process;
};

#endif 

