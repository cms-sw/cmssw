#include <iostream>
#include <fstream>
#include <sstream>
#include <inttypes.h>
#include <iomanip>
#include <cstdlib>
#include <stdio.h>
#include <string.h>
using namespace std;

const int nChs = 68;
const int nEvts = 2048;
uint16_t mem[nChs][nEvts];

/** \file
 * The TCC memory for FE data emulation takes a fixed number, 2048, of events.
 * This standalone application completes a FE emulation data file with an
 * arbitrary number of events (<=2048) in order to have the required 2048
 * events. The N initial events are repeated till having 2048 events. In
 * general the number of initial events is choosen as a divider of 2048.
 */

int main(int argc, char* argv[]){
  if((argc>=2 && ( (strcmp(argv[1],"-h")==0) || (strcmp(argv[1],"--help")==0) ))
      || argc!=3){
    cout << "Usage: recycleTccEmu infile outfile\n";
    return 1;
  }
  
  string ifilename = argv[1];
  string ofilename = argv[2];
  
  for(int iCh=0; iCh<nChs; ++iCh){
    for(int iEvts = 0; iEvts<nEvts; ++iEvts){
      mem[iCh][iEvts] = 0xFFFF;
    }
  }
  
  ifstream in(ifilename.c_str());
  int chnb;
  int bcnb;
  int val ;
  int dummy ;
  int oldLineCnt = 0;  
  
  //reads input file:
  if(in){
    while(!in.eof()) {
      in >>dec>> chnb >> bcnb >>hex>> val >> dummy ;
      mem[chnb-1][bcnb] = val&0x7FF;
      if(mem[chnb-1][bcnb]!=val){
	cout << "Invalid Et value at line " << oldLineCnt+1 << ".\n";
	exit(1);
      }
      // cout<<"Channel: "<< dec <<chnb <<", BX: "
      // << dec << bcnb << " filled with val:"<< hex<< mem[chnb-1][bcnb]
      // << dec << endl;
      ++oldLineCnt;
    }
  } else{
    cout << "Failed to open file " << ifilename << "\n";
  }

  in.close();
  ofstream out(ofilename.c_str());
  
  if(!out){
    cout << "Failed to open file '" << ofilename
	 << "' in write mode.\n";
    return 1;
  }
  
  
  bool singleOldEventCnt = true;
  int oldEventCnt[nChs];
  //fills output file:
  for(int iCh = 0; iCh<nChs; ++iCh){
    int evtcnt = 0;
    //find first not initialized events:
    while(evtcnt<nEvts && mem[iCh][evtcnt]!=0xFFFF){++evtcnt;}
    //cout << "ch " << iCh << " event count: " << evtcnt << "\n";
    oldEventCnt[iCh] = evtcnt;
    if(oldEventCnt[0]!=oldEventCnt[iCh]) singleOldEventCnt = false;
    if(evtcnt==0){
      cout << "Error: no data found for channel "<< iCh << "\n";
    }
    //clones data of channel iCh
    for(int ievt = evtcnt; ievt<nEvts; ++ievt){
      if(mem[iCh][ievt]!=0xFFFF){
	cout << "Error: memory offset of channel " << iCh
	     << " events are not contiguous.\n";
	exit(1);
      }
      mem[iCh][ievt] = mem[iCh][ievt%evtcnt];
    }
    
    for(int ievt=0; ievt<nEvts; ++ievt){
      out << iCh+1 << "\t" << ievt
	  << "\t" << hex << "0x" << setfill('0') << setw(4)
	  << mem[iCh][ievt]
	  << setfill(' ') << dec << "\t0"
	  << "\n";
    }
  }

  //warning for aperiodic case:
  if(singleOldEventCnt && (nEvts%oldEventCnt[0]!=0)){
    cout << "Warning: ouput event count (2048) is not a mulitple of input "
      "event counts\n" ;
  }
  if(!singleOldEventCnt){
    stringstream s;
    for(int iCh=0; iCh<nChs; ++iCh){
      if(nEvts%oldEventCnt[iCh]){
	s << (s.str().size()==0?"":", ") << iCh;
      }
    }
    if(s.str().size()!=0)
      cout << "Warning: ouput event count (2048) for channel"
	   << (s.str().size()>1?"s":"") << " "
	   << s.str()
	   << " is not a mulitple of input event counts\n" ;
  }
  
  if(!singleOldEventCnt){
    cout << "Info: in the input file the event count depends on the channel";
  }
}

