#ifndef TCCINPUT_H
#define TCCINPUT_H

/*\struct TCCinput
 *\description structure holding TCC input
 *\author Nuno Leonardo (CERN)
 *\date February 2007
 */

#include <iostream> 

struct TCCinput {
  
  int tower;         //tower number in SM
  int bunchCrossing; 
  unsigned input;    //11 bit value input (10 energy, 1 fg)
  
  TCCinput(int tt=0, int bx=0, unsigned val=0x0) : 
    tower(tt), bunchCrossing(bx), input(val) {
    if (input>0x7ff)  {
      std::cout << "[TCCinput] saturated value 0x" 
		<< std::hex << input << std::dec
		<< std::endl;
      input &= 0x7ff;
    }
  }; 
 
  int get_energy() const {
    return input & 0x3ff; //get bits 9:0
  }
  
  int get_fg() const {
    return input & 0x400; //get bit number 10
  }

  bool is_current (int n) const {
    return (n==bunchCrossing);
  }

  bool operator<(const TCCinput &) const;
  
  std::ostream& operator<<(std::ostream&);

};

inline 
std::ostream& TCCinput::operator<<(std::ostream& os) {
  os << " tcc input " 
     << " bx:"     << bunchCrossing 
     << " tt:"     << tower 
     << " raw:0x"  << std::hex  << input << std::dec  
     << " fg:"     << this->get_fg()
     << " energy:" << this->get_energy()
     << std::endl;
  return os;
}

inline
bool TCCinput::operator< (const TCCinput &k) const {
  return (bunchCrossing < k.bunchCrossing);
}

#endif 
