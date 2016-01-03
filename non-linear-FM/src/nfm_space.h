#ifndef NFM_SPACE_H
#define NFM_SPACE_H

#include <algorithm>
#include <assert.h>
#include <string>
#include <vector>

namespace Nfm {
namespace Internal {

/* 
 * A space describes the name of each variable of a polynom.
 * Example: in the polynom 2*X + 3*Y + 5*X*Y + 4*X^2 , the names of the 
 * variables are 'X' and 'Y'.
 */
class NfmSpace {
public:
    explicit NfmSpace(const std::vector<std::string>& names) : names_(names) {
        assert(!names_.empty());
    };
    
    explicit NfmSpace(const std::string& str);

    std::string to_string() const;
    friend std::ostream& operator<<(std::ostream& out, const NfmSpace& space);

    size_t size() const { 
        return names_.size(); 
    }

    const std::vector<std::string>& get_names() const { return names_; }

    friend bool operator==(const NfmSpace& lhs, const NfmSpace& rhs);
    friend bool operator!=(const NfmSpace& lhs, const NfmSpace& rhs);

    std::string get_name(size_t idx) { 
        assert(idx < names_.size());
        return names_[idx]; 
    }
    const std::string& get_name(size_t idx) const { 
        assert(idx < names_.size());
        const std::string& name = names_[idx];
        return name; 
    }
    std::string operator[](size_t idx) { return get_name(idx); }
    const std::string& operator[](size_t idx) const { return get_name(idx); }
    
    // Return the index of "name" in names_. If it isn't there, return -1
    int get_index(const std::string& name) const {
        const auto& iter = std::find(names_.begin(), names_.end(), name);
        if (iter != names_.end()) {
            return std::distance(names_.begin(), iter);
        }        
        return -1;
    }
    
private:
    std::vector<std::string> names_;
};

}
}

#endif
