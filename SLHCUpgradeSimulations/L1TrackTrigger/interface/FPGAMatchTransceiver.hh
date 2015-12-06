//This class implementes the projection tranceiver
#ifndef FPGAMATCHTRANSCEIVER_H
#define FPGAMATCHTRANSCEIVER_H

#include "FPGAProcessBase.hh"

using namespace std;

class FPGAMatchTransceiver:public FPGAProcessBase{

public:

  FPGAMatchTransceiver(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="matchout1"||
	output=="matchout2"||
	output=="matchout3"||
	output=="matchout4"||
	output=="matchout5"||
	output=="matchout6"||
	output=="matchout7"||
	output=="matchout8"
	){
      FPGAFullMatch* tmp=dynamic_cast<FPGAFullMatch*>(memory);
      assert(tmp!=0);
      outputmatches_.push_back(tmp);
      return;
    }
    assert(0);
  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="proj1in"||
	input=="proj2in"||
	input=="proj3in"||
	input=="proj4in"||
	input=="proj5in"||
	input=="proj6in"||
	input=="proj7in"||
	input=="proj8in"||
	input=="proj9in"||
	input=="proj10in"||
	input=="proj11in"||
	input=="proj12in"||
	input=="proj13in"||
	input=="proj14in"||
	input=="proj15in"||
	input=="proj16in"
	){
      FPGAFullMatch* tmp=dynamic_cast<FPGAFullMatch*>(memory);
      assert(tmp!=0);
      inputmatches_.push_back(tmp);
      return;
    }

    assert(0);
  }

  //this->inputmatches_ to other->outputmatches_ 
  void execute(FPGAMatchTransceiver* other){
    int count=0;
    for(unsigned int i=0;i<inputmatches_.size();i++){
      string basename=inputmatches_[i]->getName().substr(0,10);
      //cout << "input : "<<inputmatches_[i]->getName()<<" "<<basename<<endl;
      bool wrote=false;
      for(unsigned int j=0;j<other->outputmatches_.size();j++){
	//cout << "output  : "<<other->outputmatches_[j]->getName()<<endl;
	std::size_t found = other->outputmatches_[j]->getName().find(basename);
	if (found!=std::string::npos){
	  wrote=true;
	  for(unsigned int l=0;l<inputmatches_[i]->nMatches();l++){
	    other->outputmatches_[j]->addMatch(inputmatches_[i]->getMatch(l));
	  }
	  count+=inputmatches_[i]->nMatches();
	  //continue;
	}	
      }
      assert(wrote);
    }
    if (writeMatchTransceiver) {
      static ofstream out("matchtransceiver.txt");
      out << getName() << " " 
	  << count << endl;
    }
  }
  

private:

  vector<FPGAFullMatch*> inputmatches_;

  vector<FPGAFullMatch*> outputmatches_;

};

#endif
