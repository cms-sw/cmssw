#ifndef L1TWORD_H
#define L1TWORD_H

#include <iostream>
#include <assert.h>
using namespace std;

//This is the number of strips in rphi and in z for a module.
//This should be in the header of the ASCII file, but for now
//just hardcoded here.



class L1TWord{

public:

  explicit L1TWord(bool printstat=false){

    //cout << "In constructor 1"<<endl;

    setvalue(0);
    maxsize_=65535*32768;
    name_="noname";
    printstat_=printstat;
    maxvalue_=0;

  }

  L1TWord(const char* const name, bool printstat=false){

    //cout << "In constructor 2"<<endl;

    setvalue(0);
    maxsize_=65535*32768;
    name_=name;
    printstat_=printstat;
    maxvalue_=0;

  }

  explicit L1TWord(long long int maxsize, bool printstat=false){

    //cout << "In constructor 3"<<endl;

    setvalue(0);
    maxsize_=maxsize;
    name_="noname";
    printstat_=printstat;
    maxvalue_=0;

  }

  L1TWord(long long int maxsize, const char* const name, bool printstat=false){

    //cout << "In constructor 4"<<endl;

    setvalue(0);
    maxsize_=maxsize;
    name_=name;
    printstat_=printstat;
    maxvalue_=0;
  }

  L1TWord(long long int maxsize, long long int value, bool printstat=false){

    //cout << "In constructor 5"<<endl;

    setvalue(value);
    maxsize_=maxsize;
    name_="noname";
    printstat_=printstat;
    maxvalue_=0;

    assert(fabs(value)<maxsize);

  }

  L1TWord(long long int maxsize, long long int value, const char* const name,
	  bool printstat=false){

    //cout << "In constructor 6"<<endl;

    setvalue(value);
    maxsize_=maxsize;
    name_=name;
    printstat_=printstat;
    maxvalue_=0;

    assert(fabs(value)<maxsize);

  }

  ~L1TWord(){

    if (printstat_) {
      cout << name_ << " " << maxsize_ << " " << maxvalue_ << endl;
    }

  }


  L1TWord& operator=(const L1TWord& other){
    setvalue(other.value_);
    assert(fabs(value_)<maxsize_);
    return *this;
  }

  L1TWord& operator=(long long int value){
    setvalue(value);
    if (fabs(value_)>maxsize_){
      cout <<"operator= "<<name_<<" "<<maxsize_<<" "<<value<<endl;
    }
    assert(fabs(value_)<maxsize_);
    return *this;
  }

  L1TWord& operator-=(const L1TWord& other){
    setvalue(value_-other.value_);
    assert(fabs(value_)<maxsize_);
    return *this;
  }

  L1TWord& operator+=(const L1TWord& other){
    setvalue(value_+other.value_);
    if (fabs(value_)>maxsize_){
      cout << "Adding "<<name_<<"("<<value_<<")"<<" and "
	   <<other.name_<<"("<<other.value_<<")"<<endl;
    }
    assert(fabs(value_)<maxsize_);
    return *this;
  }

  L1TWord operator/(const L1TWord& other){
    //cout << "operator/:"<<value_<<" "<<other.value_<<endl;
    long long int value=value_/other.value_;
    L1TWord tmp(maxsize_,value);
    return tmp;
  }

  L1TWord operator/(long long int value){
    //cout << "operator/:"<<value_<<" "<<value<<endl;
    long long int ratio=value_/value;
    L1TWord tmp(maxsize_,ratio);
    return tmp;
  }

  L1TWord operator*(const L1TWord& other){
    long long int value=value_*other.value_;
    L1TWord tmp;
    if (fabs(value_*1.0*other.value_)>tmp.maxsize_) {
      cout << "operator*:"<<name_<<"("<<value_<<") "
	   <<other.name_<<"("<<other.value_<<")"<<endl;
    }
    assert(fabs(value)<tmp.maxsize_);
    tmp=value;
    return tmp;
  }

  L1TWord operator*(long long int value){
    long long int prod=value_*value;
    L1TWord tmp;
    if (fabs(value_*1.0*value)>tmp.maxsize_) {
      cout << "operator*(int):"<<"("<<name_<<")"<<value_<<" "<<value<<endl;
    }
    assert(fabs(prod)<tmp.maxsize_);
    tmp=prod;
    return tmp;
  }

  L1TWord operator+(const L1TWord& other){
    long long int value=value_+other.value_;
    assert(fabs(value)<maxsize_);
    L1TWord tmp(maxsize_,value);
    return tmp;
  }

  L1TWord operator+(long long int value){
    long long int sum=value_+value;
    L1TWord tmp;
    assert(fabs(sum)<tmp.maxsize_);
    tmp=sum;
    return tmp;
  }

  L1TWord operator-(const L1TWord& other){
    long long int value=value_-other.value_;
    assert(fabs(value)<maxsize_);
    L1TWord tmp(maxsize_,value);
    return tmp;
  }

  long long int value() {
    return value_;
  }

  void setName(const char* const name, bool printstat=false){
    name_=name;
    printstat_=printstat;
  }
 

private:

  void setvalue(long long int value){
    value_=value;
    if (value>maxvalue_) {
      maxvalue_=value;
    }
  }


  long long int value_;
  long long int maxsize_;

  std::string name_;
  
  bool printstat_;
  long long int maxvalue_;

};



#endif



