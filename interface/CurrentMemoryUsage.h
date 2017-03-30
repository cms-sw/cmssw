#ifndef CurrentMemoryUsage_h
#define CurrentMemoryUsage_h

#include <iostream>
#include <string>
#include <sys/time.h>
#include <sys/resource.h> 

void PrintCurrentMemoryUsage(std::string comment);

static long Previousru_idrss = 0; 
static long Previousru_isrss = 0; 
static long Previousru_nswap = 0; 


#endif
