
#include <time.h>

typedef struct {
  int running; /* boolean */
  double last_time;
  double total;

} *Stopwatch, Stopwatch_struct;

double seconds();

void Stopwtach_reset(Stopwatch Q);

Stopwatch new_Stopwatch(void);
void Stopwatch_delete(Stopwatch S);
void Stopwatch_start(Stopwatch Q);
void Stopwatch_resume(Stopwatch Q);
void Stopwatch_stop(Stopwatch Q);
double Stopwatch_read(Stopwatch Q);
