#include <vector>
#include <algorithm>
#include <memory>
#include "TrackingTools/PatternTools/interface/bqueue.h"
#include<iostream>
#include<cassert>

typedef cmsutils::bqueue<std::unique_ptr<int>> Cont;

inline
void verifySeq(Cont const & cont, int incr=0) {
  int n=cont.size();
  for ( auto const & item : cont) {
    assert( (*item)-incr == --n);
  }
  assert(n==0);
}


int main() {

  Cont cont, cont1, cont2;
  assert(cont.size()==0);
  verifySeq(cont);
  assert(cont.begin()==cont.end());


  constexpr int v[] = {0,1,2,3,4,5,6,7,8,9};
  constexpr int v1[] = {10,11,12,13,14,15,16,17,18,19};
  constexpr int N=sizeof(v)/sizeof(int);
  constexpr int N1=sizeof(v1)/sizeof(int);

  for (auto i: v) cont.push_back(Cont::value_type(new int(i)));
  for (auto i: v) cont1.emplace_back(new int(i));
  for (auto i: v1) cont2.emplace_back(new int(i));
  
  assert(cont.size()==N);
  assert(cont1.size()==N);
  assert(cont2.size()==N1);
  verifySeq(cont);
  verifySeq(cont1);
  verifySeq(cont2,10);
  
  // copy
  assert(!cont.shared());
  Cont cont3(cont);
  assert(cont.size()==cont3.size());
  assert(cont.shared());
  assert(cont3.shared());
  verifySeq(cont3);
  // add
  cont.push_back(Cont::value_type(new int(10)));
  assert((*cont.back())==10);
  assert((*cont3.back())==9);
  assert(cont.shared());
  assert(cont3.shared());
  verifySeq(cont);
  cont3.clear();
  assert(!cont.shared());
  assert(cont.size()==N+1);
  verifySeq(cont);

  // join
  cont1.join(cont2);
  assert(cont1.size()==N+N1);
  assert(cont2.size()==0);
  assert(cont2.begin()==cont2.end());
  verifySeq(cont1);
  verifySeq(cont2);
  assert(!cont1.shared());
  // add
  cont1.push_back(Cont::value_type(new int(20)));
  assert((*cont1.back())==20);
  verifySeq(cont1);


  Cont h, t1;
  for (auto i: v) h.emplace_back(new int(i));
  for (auto i: v1) t1.emplace_back(new int(i));
  Cont t2(t1);
  assert(!h.shared());
  assert(t1.shared());
  assert(t2.shared());
  // h.join(t1); //  bombs..
  t2.clear();


  cont.clear();
  assert(cont.size()==0);
  verifySeq(cont);
  assert(cont.begin()==cont.end());

  return cont.size();

}
