// The glibc malloc hooks removed in 2.34 and Xorg's ErrorF, as harmless definitions, for testing
// whether a driver needs them to exist.
#include <stddef.h>
void *__malloc_hook = NULL;
void *__realloc_hook = NULL;
void *__free_hook = NULL;
void *__memalign_hook = NULL;
void ErrorF(const char *fmt, ...) { (void)fmt; }
