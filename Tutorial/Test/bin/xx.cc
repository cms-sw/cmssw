
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

int main (int argc, char* argv[]) {

   FILE* fp=    fopen(argv[1], "a+");
   if (fp == NULL) {
      printf("CANT topen file !!\n");
   }
   printf("test test \n");
   fprintf(fp, "test test \n");

   fflush(fp);
   // fclose(fp);

   return 0;
}
