// File: MixingWorkerBase.cc
// Description:  see MixingWorkerBase.h
// Author:  Ursula Berthon, LLR Palaiseau
//
//--------------------------------------------

#include "MixingWorkerBase.h"

namespace edm
{

  // Virtual destructor needed so MixingModule can delete the list
  // of MixingWorkers without knowing their exact type.
  MixingWorkerBase::~MixingWorkerBase() { }  

}//edm
