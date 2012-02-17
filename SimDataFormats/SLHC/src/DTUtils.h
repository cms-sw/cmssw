#ifdef SLHC_DT_TRK_DFENABLE
#ifndef __UUTILS_
#define __UUTILS_


#include <vector>

static const double pi_greca = 3.14159265358979323846;

// to write in binary format
char* biny (int Value, bool truncated_on_the_left = false , size_t wordSize = 32);

#endif

#endif
