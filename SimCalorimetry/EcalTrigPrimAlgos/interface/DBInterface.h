#ifndef DB_INTERFACE_H
#define DB_INTERFACE_H

#include <vector>
#include <map>

class DBInterface {
  
 public:
  DBInterface() {;}
  DBInterface(std::string dbFileEB,std::string dbFileEE);
   std::vector<unsigned int> getTowerParameters(int SM, int towerInSM, bool print = false) ;
  std::vector<unsigned int> getStripParameters(int SM, int towerInSM, int stripInTower, bool print = false) ;
  std::vector<unsigned int> getXtalParameters(int SM, int towerInSM, int stripInTower, int xtalInStrip, bool print = false) ;

 private:
  std::vector<int> getRange(int smNb, int towerNbInSm, int stripNbInTower=0, int xtalNbInStrip=0) ;

  std::map <int, std::vector<unsigned int> > towerParam_ ;
  std::map <int, std::vector<unsigned int> > stripParam_ ;
  std::map <int, std::vector<unsigned int> > xtalParam_ ;

};


#endif
