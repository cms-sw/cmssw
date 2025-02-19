#ifndef ReadED_Mapping_h
#define ReadED_Mapping_h
#include <string>
#include <map>

struct chamstrip{
  region;
  ringwheel;
  sector;
  subsector;
  station;
  layer;
  roll;
  strip;
  
}

class Mapping{
 public:
  Mapping();
  Mapping(int wheel, int sector);
  ~Mapping();
  
  chamstrip stripind(std::string lbname, int channel);
 private:
  int w;
  int s;
  std::map<std::string, std::map<int,int> > maps;
};

#endif
