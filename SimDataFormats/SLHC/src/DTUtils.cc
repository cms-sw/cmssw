#ifdef SLHC_DT_TRK_DFENABLE
#include <iostream>
#include <sstream>
#include <math.h>
#include "DTUtils.h"

using namespace std;


char* biny (int Value, bool truncated_on_the_left , size_t wordSize) {
  const size_t WordSize = wordSize + 1;
  //unsigned int Mask = 0X80000000;

  unsigned int Mask = 1;
  for(size_t i=0; i<(wordSize-1); i++)
    Mask <<= 1;

  int Bit;
  char* Text = new char[WordSize];
  Bit = 0;
  while (Mask != 0) {
    Text [Bit] = ((Mask & Value) == 0) ? '0' : '1';
    Mask >>= 1;
    ++Bit;
  }
  Text [Bit] = 0; // termination character

  if(!truncated_on_the_left)
    return Text;

  for(size_t i=0; i<WordSize; i++)
    cout << Text[i] << flush;
  cout << endl;

  size_t zeros_on_left = 0;
  while(Text[zeros_on_left] == '0') {
    //cout << zeros_on_left << ")\t" << Text[zeros_on_left] << endl;
    zeros_on_left += 1;
  }
  const size_t size = WordSize - zeros_on_left;
  char* FinalText = new char[size];
  for(size_t i=0; i<size; i++)
    FinalText[i] = Text[zeros_on_left + i];
  FinalText[size] = 0;
  return FinalText;
}

#endif
