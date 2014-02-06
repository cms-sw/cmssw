#include "TauAnalysis/MCEmbeddingTools/interface/extraPythia.h"

void call_pyexec()
{ 
  pyexec(); 
}

void call_py1ent(int ip,int flavour,double energy,double theta,double phi)
{ 
  py1ent(ip,flavour,energy,theta,phi); 
}

void call_pyedit(int medit)
{ 
  pyedit(medit);
}
