#ifndef DB_INTERFACE_H
#define DB_INTERFACE_H

#include <vector>
#include <map>
#include <string>

class DBInterface {
  
 public:
  DBInterface() {;}
  DBInterface(std::string dbFileEB,std::string dbFileEE);
  std::vector<unsigned int> getTowerParameters(int SM, int towerInSM) ;
  std::vector<unsigned int> getStripParameters(int SM, int towerInSM, int stripInTower) ;
  std::vector<unsigned int> getXtalParameters(int SM, int towerInSM, int stripInTower, int xtalInStrip) ;

 private:
  std::map <int, std::vector<unsigned int> > towerParam_ ;
  std::map <int, std::vector<unsigned int> > stripParam_ ;
  std::map <int, std::vector<unsigned int> > xtalParam_ ;

};


#endif
