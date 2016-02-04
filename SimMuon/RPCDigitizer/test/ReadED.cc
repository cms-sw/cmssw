#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <bitset>

int main(int argc, char** argv){
  std::bitset<4> a;
  a=1;
  std::cout <<a<<std::endl;
  a<<=1;
  std::cout <<a<<std::endl;
  a<<=2;
  a|=1;
  std::cout <<a<<std::endl;

  if(argc != 2) return -1;
  std::string filename=argv[1];
  std::ifstream inpf(filename.c_str());
  std::string buf;
  std::vector<std::string> lbname;
  bool lb=false;
  bool bx=true;
  int ilb=0;
  std::map<int, std::map<std::string,std::bitset<96> > > event;
  std::map<std::string, std::bitset<96> > lbdump;
  int ibx = -1;
  do{
    buf.clear();
    inpf >>buf;
    if (buf=="lb"){
      lb=true;
    } else if (lb){
      lb = false;
      lbname.push_back(buf);
    } else if (!buf.empty()){
      if (bx) {
	bx = false;
	ilb=0;
	std::stringstream os;
	os<<buf;
	os>>ibx;
	std::cout <<" bx = "<<ibx<<" |"<<buf<<"|"<<std::endl;
	lbdump.clear();
      }else{
	ilb++;
	if (ilb==lbname.size())	  
	  bx = true;
	unsigned long icont1,icont2,icont3,icont4;
	{
	  std::stringstream os;
	  os<<buf.substr(0,6);
	  os>>std::hex>>icont1;	  
	}
	{
	  std::stringstream os;
	  os<<buf.substr(6,6);
	  os>>std::hex>>icont2;	  
	}
	{
	  std::stringstream os;
	  os<<buf.substr(12,6);
	  os>>std::hex>>icont3;	  
	}
	{
	  std::stringstream os;
	  os<<buf.substr(18,6);
	  os>>std::hex>>icont4;	  
	}

	std::bitset<96> cont(icont1);
	cont <<=24;
	cont |= icont2;
	cont <<=24;
	cont |= icont3;
	cont <<=24;
	cont |= icont4;

	if (cont.count()>0){
	  lbdump[lbname[ilb-1]]=cont;
	  std::cout <<cont.count()<<" Content of "
		    <<lbname[ilb-1]<<" = "<<buf<<std::endl;
	  std::cout <<"Content of "<<lbname[ilb-1]<<" = "
		    <<buf.substr(0,6)<<" "<<buf.substr(6,6)
		    <<" "<<buf.substr(12,6)<<" "<<buf.substr(18,6)<<std::endl;
	  std::cout <<"Content of "<<lbname[ilb-1]<<" = "<<std::hex
		    <<icont1<<" "<<icont2<<" "<<icont3<<" "<<icont4
		    <<std::dec<<std::endl;
	  std::cout <<"Content of "<<lbname[ilb-1]<<" = "<<cont<<std::endl;
	}
	if (bx){
	  if (lbdump.size()>0){
	    std::cout <<"good bx "<<ibx<<std::endl;
	    event[ibx]=lbdump;
	  }
	}
      }
    }  
  }while(inpf.good() && !buf.empty());


  std::cout <<"done!"<<std::endl;

  for (std::map<int, std::map<std::string, std::bitset<96> > >::iterator ix=
	 event.begin(); ix!=event.end();ix++){

    std::cout <<"Bunch Crossing "<<ix->first<<std::endl;
    for (std::map<std::string, std::bitset<96> >::iterator ib=
	   ix->second.begin(); ib != ix->second.end(); ib++){
      if (ib->second.count()>0){
	std::cout <<"Board "<<ib->first<<std::endl;
	for (int i=0;i<96;i++){
	  if (ib->second[i]>0){
	    std::cout <<"channel = "<<i<<std::endl;
	  }
	}
      }
    }

  }
       

  
  
}
