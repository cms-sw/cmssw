#include "Utilities/General/interface/stringTools.h"

using std::string;

int replace(string & input, const string&gone, const string& it, bool multiple) {
  int n=0;
  size_t i = input.find(gone,0);
  while(i!=string::npos) {
      n++;
      input.replace(i,gone.size(),it);
      i = input.find(gone,i+(multiple ? 0 : it.size()));
  }
  return n;
}

void strip(std::string & input, const std::string& blanks) {
  size_t b = input.find_first_not_of(blanks);
  if (b==std::string::npos) { input.clear(); return;}
  size_t e = input.find_last_not_of(blanks);
  input = input.substr(b,e-b+1);
}

int replaceRange(string & input, const string&first, const string& last, const string& it, bool multiple) {
  int n=0;
  size_t i = input.find(first,0);
  while(i!=string::npos) {
    size_t e = input.find(last,i);
    if (e!=string::npos) {
      n++;
      input.replace(i,e+last.size()-i,it);
      i = input.find(first,i+(multiple ? 0 : it.size()));
    } else break;
  }
  return n;
}
