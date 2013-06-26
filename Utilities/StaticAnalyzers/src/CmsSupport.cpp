//===--- CmsSupport.cpp - Provides support functions ------------*- C++ -*-===//
//
// by Shahzad Malik MUZAFFAR [ Shahzad.Malik.MUZAFFAR@cern.ch ]
//
//===----------------------------------------------------------------------===//

#include "CmsSupport.h"
#include <cstdlib>
#include <cstring>
using namespace clangcms;

bool support::isCmsLocalFile(const char* file)
{
  static char* LocalDir=0;
  static int DirLen=-1;
  if (DirLen==-1)
  {
    DirLen=0;
    LocalDir = getenv ("LOCALRT");
    if (LocalDir!=NULL) DirLen=strlen(LocalDir);
  }
  if ((DirLen==0) || (strncmp(file,LocalDir,DirLen)!=0) || (strncmp(&file[DirLen],"/src/",5)!=0)) return false;
  return true;
}
