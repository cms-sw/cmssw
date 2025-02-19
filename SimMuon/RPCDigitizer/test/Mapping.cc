#include "Mapping.h"
#include <vector>
#include <fstream>

Mapping::Mapping()
{

}


Mapping::Mapping(int wheel, int sector) : w(wheel), s(sector)
{
  std::map<std::string, std::vector<int>> lbnames;

  std::vector<int> missing_channel;
  missing_channel.push_back(0);
  lbnames["RB1IN"]  = missing_channel;

  missing_channel.clear();
  missing_channel.push_back(0);
  missing_channel.push_back(8);
  lbnames["RB1OUT"] = missing_channel;  

  missing_channel.clear();
  missing_channel.push_back(0);
  lbnames["RB22IN"] = missing channel;

  missing_channel.clear();
  missing_channel.push_back(0);
  missing_channel.push_back(1);
  lbnames["RB22OUT"]=missing_channel;

  missing_channel.clear();
  missing_channel.push_back(0);
  lbnames["RB23IN"] =missing_channel;


  missing_channel.clear();
  missing_channel.push_back(0);
  missing_channel.push_back(1);
  lbnames["RB23OUT"]=missing_channel;

  missing_channel.clear();
  missing_channel.push_back(0);
  missing_channel.push_back(15);
  lbnames["RB3"]    = missing_channel;
  if (wheel==1 && sector==10){
    missing_channel.clear();
    missing_channel.push_back(0);
    missing_channel.push_back(1);
    missing_channel.push_back(14);
    missing_channel.push_back(15);
    lbnames["RB4+"]  = missing_channel;
    lbnames["RB4-"]  = missing_channel; 
  }
  
  for(std::map<std::string, std::string>::iterator i=lbnames.begin();
      i!=lbnames.end(); i++){

    for (int ich=0;ich<96;ich++){
      maps[i->first]=
    
  }

}


chamstrip
stripind(std::string lbname, int channel){


}




  
