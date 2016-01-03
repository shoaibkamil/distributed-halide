#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

#include <isl/set.h>

#include "nfm_isl_interface.h"

#include "nfm_debug.h"

namespace Nfm {
namespace Internal {

using std::ostringstream;
using std::string;

namespace {

isl_stat extract_basic_set_from_set(isl_basic_set *bset, void *bset_list) {
    isl_bset_list **list = (isl_bset_list **) bset_list;

    if (*list == NULL) {
        isl_bset_list *node = (isl_bset_list *) malloc(sizeof(isl_bset_list));
        node->bset = bset;
        node->next = NULL;
        *list = node;
    } else {
        isl_bset_list *node = (isl_bset_list *) malloc(sizeof(isl_bset_list));
        node->bset = bset;
        node->next = (*list);
        *list = node;
    }
    return isl_stat_ok;
}

}

/* Return the list of basic sets in a set.  */
isl_bset_list *isl_set_get_bsets_list(isl_set *set) {
    assert(set != NULL);
    isl_bset_list *list = NULL;
    isl_set_foreach_basic_set(set, extract_basic_set_from_set, &list);
    return list;
}

void isl_bset_list_dump(isl_bset_list *list) {
    isl_bset_list *node = list;
    while (node != NULL) {
        isl_basic_set_dump(node->bset);
        node = node->next;
    }
}

void isl_bset_list_free(isl_bset_list *list) {
    isl_bset_list *node = list;
    isl_bset_list *next_node;

    while (node != NULL) {
        isl_basic_set_free(node->bset);
        next_node = node->next;
        free(node);
        node = next_node;
    }
}

}
}
