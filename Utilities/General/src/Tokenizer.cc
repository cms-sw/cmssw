#include "Utilities/General/interface/Tokenizer.h"

Tokenizer::Tokenizer(const std::string & sep, const std::string & input, bool alsoempty) {
  size_type i=0, j=0;
  while( (j=input.find(sep,i))!=std::string::npos) {
    if (alsoempty || (j>i) ) push_back(input.substr(i,j-i));
    i = j+sep.size();
  }
  if (alsoempty || (i<input.size()) ) push_back(input.substr(i));
}

void Tokenizer::join(std::string & out, const std::string & sep, bool alsoempty) const {
    for (super::const_iterator p=begin(); p!=end(); p++)
      if (alsoempty || (!(*p).empty()) ) { out+=*p; out+=sep;}
}
