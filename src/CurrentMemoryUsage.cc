#include "CurrentMemoryUsage.h"
#include <stdio.h>
#include <stdlib.h>


void PrintCurrentMemoryUsage(std::string comment)
{
//  struct rusage usage;
//  std::cout<<comment<<", memory use info"<<std::endl;
//  system("ps aux|head -1;ps aux|grep FindAppro");
  
/*  if(getrusage(RUSAGE_SELF, &usage))
  {
    std::cout<<"Memory information not available"<<std::endl;
  }
  else
  {
    std::cout<<comment<<", memory use info"<<std::endl;
    std::cout<<"integral unshared data size = "<<usage.ru_idrss<<"  increased = "<<(usage.ru_idrss-Previousru_idrss)<<std::endl;
    std::cout<<"integral unshared stack size = "<<usage.ru_isrss<<"  increased = "<<(usage.ru_isrss-Previousru_isrss)<<std::endl;
    std::cout<<"swaps = "<<usage.ru_nswap<<"  increased = "<<(usage.ru_nswap-Previousru_nswap)<<std::endl;
  
    Previousru_idrss = usage.ru_idrss;
    Previousru_isrss = usage.ru_isrss;
    Previousru_nswap = usage.ru_nswap;
  }*/
}

