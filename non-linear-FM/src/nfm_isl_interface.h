#ifndef NFM_ISL_INTERFACE_H
#define NFM_ISL_INTERFACE_H

#include <isl/set.h>

#include "nfm.h"

namespace Nfm {
namespace Internal {

typedef struct isl_bset_list {
    isl_basic_set *bset;
    struct isl_bset_list *next;
} isl_bset_list;

/* Return the list of basic sets in a set.  */
isl_bset_list *isl_set_get_bsets_list(isl_set *set);

void isl_bset_list_dump(isl_bset_list *list);

void isl_bset_list_free(isl_bset_list *list);

}
}

#endif
