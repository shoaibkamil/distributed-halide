#ifndef NFM_DEBUG_H
#define NFM_DEBUG_H

namespace Nfm {
namespace Internal {

/* Set to 1 for basic debugging.  */
#define NFM_DEBUG_LEVEL_1 0

/* Set to 1 for more verbose debugging.  */
#define NFM_DEBUG_LEVEL_2 0

#define IF_DEBUG(x) if(NFM_DEBUG_LEVEL_1) {x;}
#define IF_DEBUG2(x) if(NFM_DEBUG_LEVEL_2) {x;}

}
}

#endif
