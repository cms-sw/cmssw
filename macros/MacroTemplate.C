#include "L1Ntuple.h"
#include "hist.C"
#include "Style.C"

// --------------------------------------------------------------------
//                       MacroTemplate macro definition
// --------------------------------------------------------------------

class MacroTemplate : public L1Ntuple
{
  public :

    //constructor    
    MacroTemplate(std::string filename) : L1Ntuple(filename) {}
    MacroTemplate() {}
    ~MacroTemplate() {}

    //main function macro : arguments can be adpated to your need
    void run(Long64_t nevents);

  private : 

    //your private methods can be declared here
};


// --------------------------------------------------------------------
//                             run function 
// --------------------------------------------------------------------
void MacroTemplate::run(Long64_t nevents)
{
  //load TDR style
  setTDRStyle();

  //number of events to process
  if (nevents==-1 || nevents>GetEntries()) nevents=GetEntries();
  std::cout << nevents << " to process ..." << std::endl;

  //loop over the events
  for (Long64_t i=0; i<nevents; i++)
    {
      //load the i-th event 
      Long64_t ientry = LoadTree(i); if (ientry < 0) break;
      GetEntry(i);

      //process progress
      if(i!=0 && (i%10000)==0) 
	std::cout << "- processing event " << i << "\r" << std::flush;

      //write your code here

    }
}
