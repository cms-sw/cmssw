#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>

#include "dablooms.h"

#define CAPACITY 5000
#define ERROR_RATE 0.0002

static void chomp_line(char *word)
{
    char *p;
    if ((p = strchr(word, '\r'))) {
        *p = '\0';
    }
    if ((p = strchr(word, '\n'))) {
        *p = '\0';
    }
}

int generate_bloom_filter(const char *bloom_file, const char *words_file)
{
    FILE *fp;
    char word[1024];
    scaling_bloom_t *bloom;
    int i;
    
    if (!(bloom = new_scaling_bloom(CAPACITY, ERROR_RATE, bloom_file))) {
        fprintf(stderr, "ERROR: Could not create bloom filter\n");
        return EXIT_FAILURE;
    }
    
    if (!(fp = fopen(words_file, "r"))) {
        fprintf(stderr, "ERROR: Could not open words file\n");
        return EXIT_FAILURE;
    }
    
    for (i = 0; fgets(word, sizeof(word), fp); i++) {
        chomp_line(word);
        scaling_bloom_add(bloom, word, strlen(word), i);
    }
    
    int result = bitmap_flush(bloom->bitmap);
    
    return result;
}


int main(int argc, char *argv[])
{
    printf("** dablooms version: %s\n", dablooms_version());
    
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <bloom_file> <words_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int result = generate_bloom_filter(argv[1], argv[2]);
    return result;
}
