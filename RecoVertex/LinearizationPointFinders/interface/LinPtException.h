#ifndef LinPtException_H
#define LinPtException_H

#include "FWCore/Utilities/interface/Exception.h"

  /** 
   *  A class that wraps cms::Exception by deriving from it.
   */

class LinPtException : public cms::Exception
{
public:
  LinPtException( const char * reason ) : cms::Exception ( reason ) {};
};

#endif
