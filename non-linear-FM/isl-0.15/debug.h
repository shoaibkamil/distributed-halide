#ifndef __H_DEBUG
#define __H_DEBUG

#include <stdio.h>

/* Set to 1 for basic debugging.  */
#define DEBUG_LEVEL_1 1

/* Set to 1 for more verbose debugging.  */
#define DEBUG_LEVEL_2 0

#define IF_DEBUG(x) if(DEBUG_LEVEL_1) {x; fflush(stdout);}
#define IF_DEBUG2(x) if(DEBUG_LEVEL_2) {x; fflush(stdout);}

#endif
