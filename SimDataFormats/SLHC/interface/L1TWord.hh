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

  L1TWord(){

    value_=0;
    maxsize_=65535*32768;

  }

  explicit L1TWord(long long int maxsize){

    value_=0;
    maxsize_=maxsize;

  }

  L1TWord(long long int maxsize, long long int value){

    value_=value;
    maxsize_=maxsize;

    assert(fabs(value)<maxsize);

  }

  L1TWord& operator=(const L1TWord& other){
    value_=other.value_;
    assert(fabs(value_)<maxsize_);
    return *this;
  }

  L1TWord& operator=(long long int value){
    value_=value;
    assert(fabs(value_)<maxsize_);
    return *this;
  }

  L1TWord& operator-=(const L1TWord& other){
    value_-=other.value_;
    assert(fabs(value_)<maxsize_);
    return *this;
  }

  L1TWord& operator+=(const L1TWord& other){
    value_+=other.value_;
    if (fabs(value_)>maxsize_){
      cout << "Adding "<<value_<<" and "<<other.value_<<endl;
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
    if (fabs(value_*1.0*other.value_)>maxsize_) {
      cout << "operator*:"<<value_<<" "<<other.value_<<endl;
    }
    assert(fabs(value)<maxsize_);
    L1TWord tmp(maxsize_,value);
    return tmp;
  }

  L1TWord operator*(long long int value){
    long long int prod=value_*value;
    if (fabs(value_*1.0*value)>maxsize_) {
      cout << "operator*(int):"<<value_<<" "<<value<<endl;
    }
    assert(fabs(prod)<maxsize_);
    L1TWord tmp(maxsize_,prod);
    return tmp;
  }

  L1TWord operator+(const L1TWord& other){
    long long int value=value_+other.value_;
    assert(fabs(value)<maxsize_);
    L1TWord tmp(maxsize_,value);
    return tmp;
  }

  long long int value() {
    return value_;
  }
  


private:

  long long int value_;
  long long int maxsize_;

};



#endif



