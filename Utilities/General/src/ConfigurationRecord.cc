#define CMS_NOTUSING_STD

#include "Utilities/General/interface/ConfigurationRecord.h"
#include "Utilities/General/interface/envUtil.h"
#include "Utilities/General/interface/ioutils.h"
#include "Utilities/General/interface/stringTools.h"
#include "Utilities/General/interface/Tokenizer.h"

#include <istream>
#include "Utilities/General/interface/CMSexception.h"
#include "Utilities/General/interface/GenUtilVerbosity.h"

#include <algorithm>
#include <iterator>

using std::string;

const ConfigurationRecord::Keys ConfigurationRecord::defKeys;


ConfigurationRecord::Keys::Keys() : separator_("=") {
  modifiers_.push_back("+");
  comments_.push_back("//");
  comments_.push_back("#");
}

ConfigurationRecord::Keys::Keys(const std::string & isep) : separator_(isep) {
  modifiers_.push_back("+");
  comments_.push_back("//");
  comments_.push_back("#");
}

ConfigurationRecord::Keys::Keys(const std::string & isep, const std::vector<std::string> & imod, const std::vector<std::string> & icomm) :
  separator_(isep), modifiers_(imod), comments_(icomm){}



ConfigurationRecord::ConfigurationRecord(const Keys & ik, bool evEnVar): 
  keys(ik), evalEnVar_(evEnVar) {}

ConfigurationRecord::ConfigurationRecord(std::istream & input, const Keys & ik, bool evEnVar) : 
  keys(ik), evalEnVar_(evEnVar)
{
  add(input);
  parse();
}

ConfigurationRecord::ConfigurationRecord(const std::string & isource, const Keys & ik, bool evEnVar) : 
  source_(isource), keys(ik), evalEnVar_(evEnVar)
{parse();}

void ConfigurationRecord::add(const std::string & isource) {
  if (isource.empty()) return;
  source_ += isource;
  source_ +="\n";
}
void ConfigurationRecord::add(std::istream & input) {
  input.unsetf( std::ios::skipws );
  std::istream_iterator<char > sbegin(input),send;
  std::copy(sbegin,send,inserter(source_,source_.end()));
  if (source_.empty()) return;
  source_ +="\n";
}



// does not use ospace anymore  VI 22/11/2000
void ConfigurationRecord::parse(){
  using std::string;

  if (source_.empty()) return;
  // cout << "\nSOURCE:" << endl;
  // cout << source_ << endl;

  std::vector<std::string> asisText;
  // strip asis text 
  {
    size_t i = source_.find("@{",0);
    while(i!=string::npos) {
      size_t e = source_.find("}@",i);
      if (e!=string::npos) {
	string asis  = source_.substr(i+2,e-i-2);
	string placeOlder("@"); placeOlder+=toa()(asisText.size());
	asisText.push_back(asis);
	source_.replace(i,e+2-i,placeOlder);
	i = source_.find("@{",i);
      } else throw cms::Exception("ConfigurationRecord: unmatched @{");
    }
  }

   // evaluate env var if required
  if (evalEnVar_) {
    size_t i = source_.find("${",0);
    while(i!=string::npos) {
      size_t e = source_.find("}",i);
      if (e!=string::npos) {
	string evarN  = source_.substr(i+2,e-i-2);
 	string evarV;
	if (!evarN.empty()) evarV = envUtil(evarN.c_str()).getEnv();
	source_.replace(i,e+1-i,evarV);
	i = source_.find("${",i);
      } else throw cms::Exception("ConfigurationRecord: unmatched ${");
    }
  }  

  // cout << "special done" << endl;

  // clean source from comments

  // C stile comments (not yet implemented...)

  //inline comments
  for (Keys::iter p=keys.comments().begin();p!=keys.comments().end();p++) {
    /* int ic = */ replaceRange(source_,*p,"\n","\n");
  }
  
  //  cout << "comment done" << endl;

  // remove multiple "blanks"
  replace(source_,"  "," ",true);
  // replace tabs
  replace(source_,"\t"," ",false);
  // remove blank lines
  replace(source_," \n","\n",true);
  replace(source_,"\n  ","\n",true);
  replace(source_,"\n\n","\n",true);
  
  // cout << "blanks done" << endl;


  // find KEY separator ... and tokenize

  // get rid of blanks and cr before separator and modifier
  {
    string expr(" "); expr += keys.separator();
    replace(source_,expr,keys.separator(),false);
  }
  {
    string expr("\n"); expr += keys.separator();
    replace(source_,expr,keys.separator(),false);
  }

  for (Keys::iter p=keys.modifiers().begin();p!=keys.modifiers().end();p++) {

    {
      string expr(" "); expr += *p;
      replace(source_,expr,*p,false);
    }
    {
      string expr("\n"); expr += *p;
      replace(source_,expr,*p,false);
    }

  }

  // cout << "\nSOURCE:" << endl;
  // cout << source_ <<"-------"<< endl << endl;

  // get rid of leading and trailing blanks and comments
  {
    size_t i = source_.find_first_not_of(string(" \n/#"),0);
    if (i!=string::npos) source_.erase(0,i);
    i = source_.find_last_not_of(string(" \n/#"),string::npos);
    if (i!=string::npos) source_.resize(i+1);
  }

  //  cout << "\nSOURCE:" << endl;
  //cout << source_ <<"-------"<< endl << endl;


  // if nothing left return...
  if (source_.empty()) return;


  // add a special separator before KEY
  string sillySep("^^^");
  {
    size_t i = source_.find(keys.separator(),0); // skip first
    i = source_.find(keys.separator(),i+keys.separator().size());
    while(i!=string::npos) {
      size_t e = source_.find_last_of(string(" \n"),i);
      if (e==string::npos) 
	throw cms::Exception("ConfigurationRecord: wrong parsing of configuration");
      else e++;
      source_.insert(e,sillySep);
      i = source_.find(keys.separator(),i+3+keys.separator().size());
    } 
  }
  
  // get rid of blanks and cr after separator
  {
    string expr = keys.separator() + " ";
    replace(source_,expr,keys.separator(),true);
    expr = keys.separator() + "\n";
    replace(source_,expr,keys.separator(),true);
  }
  
  // cout << "\nSOURCE:" << endl;
  // cout << source_ << endl << endl;
  
  // if nothng left return...
  if (source_.empty()) return;

  // tokenize and insert in the dictionary
  Tokenizer tokens(sillySep, source_);
  // cout << "\nfound " << tokens.size() << " tokens " << endl;
  std::vector<string>::const_iterator e= tokens.end();
  for (std::vector<string>::const_iterator p= tokens.begin();p!=e; ++p) {
    // cout << (*p) << endl;
    string::size_type pos = (*p).find(keys.separator(),0);
    if (pos==string::npos) {
      if (!GenUtilVerbosity::silent())
	GenUtil::cout << "ConfigurationRecord: wrong parsing of token " << *p << std::endl;
      continue;
    }
    string key((*p),0,pos);
    // replace asis
    replace(key," @","@",true);
    if (key[0]=='@') {
      int i = ato<int>()(key.substr(1,string::npos).c_str());
      key = asisText[i];
    }
    string token((*p),pos+1,string::npos);
    if (!token.empty()) {
      // get rid of trailing blanks and \n
      replace(token,"  "," ",true);
      replace(token," \n","\n",false);
      replace(token,"\n ","\n",false);
      // replace(token,"\n","",false);
      if ((!token.empty())&&token[token.size()-1]=='\n') token.resize(token.size()-1);
      // replace asis
      replace(token," @","@",true);
      if (token[0]=='@') {
	int i = ato<int>()(token.substr(1,string::npos).c_str());
	token = asisText[i];
      }
    }
    // cout << key << " -> |" << token <<"|"<<endl;
    dict_.push_back(DictEl(key,token));
  }
  
}
  
void ConfigurationRecord::dump() const {
  for (DictCI p=dict_.begin(); p!=dict_.end(); ++p)
    GenUtil::cout << (*p).first << " => " << (*p).second << std::endl;
} 


