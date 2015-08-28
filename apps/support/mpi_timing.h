#ifndef _MPI_TIMING_H
#define _MPI_TIMING_H

#include <cstdio>
#include <cassert>
#include <algorithm>
#include <limits>
#include <vector>
#include <sys/time.h>
#include "mpi.h"

using std::numeric_limits;
using std::vector;

class MPITiming {
public:
    typedef enum {Min, Max, Median, Mean} Stat;

    MPITiming(MPI_Comm c) : _comm(c) {
        MPI_Comm_rank(_comm, &_rank);
        MPI_Comm_size(_comm, &_numprocs);
        _reduced = -1.0;
        _gathered = -1.0;
    }

    inline void barrier() const {
        MPI_Barrier(_comm);
    }

    void report() const {
        if (_rank == 0) {
            printf("Timing: <%d> ranks <%.3f> seconds\n", _numprocs, gathered());
        }
    }

    void record(float sec) {
        _timings.push_back(sec);
    }

    // Reduce all recorded values by the given statistic.
    void reduce(Stat stat) {
        _reduced = compute_stat(_timings, stat);
        _timings.clear();
    }

    // Gather reduced values from all ranks to rank 0, and reduce the
    // gathered values by the given statistic.
    void gather(Stat stat) {
        const int tag = 0;
        if (_rank == 0) {
            vector<float> results;
            results.push_back(_reduced);
            for (int i = 1; i < _numprocs; i++) {
                float t;
                MPI_Recv(&t, 1, MPI_FLOAT, i, tag, _comm, MPI_STATUS_IGNORE);
                results.push_back(t);
            }
            _gathered = compute_stat(results, stat);
        } else {
            MPI_Send(&_reduced, 1, MPI_FLOAT, 0, tag, _comm);
        }
    }

    // Return the gathered result.
    float gathered() const {
        return _gathered;
    }
private:
    int _rank, _numprocs;
    MPI_Comm _comm;
    vector<float> _timings;
    float _reduced;
    float _gathered;

    float compute_stat(const vector<float> &values, Stat stat) const {
        switch (stat) {
        case Min:
            return compute_min(values);
        case Max:
            return compute_max(values);
        case Median:
            return compute_median(values);
        case Mean:
            return compute_mean(values);
        }
        assert(false && "Unknown statistic.");
        return 0.0;
    }

    float compute_min(const vector<float> &values) const {
        float min = numeric_limits<float>::max();
        for (float v : values) {
            min = v <= min ? v : min;
        }
        return min;
    }

    float compute_max(const vector<float> &values) const {
        float max = -numeric_limits<float>::max();
        for (float v : values) {
            max = v >= max ? v : max;
        }
        return max;
    }

    float compute_median(const vector<float> &values) const {
        float median = 0;
        vector<float> tmp(values.begin(), values.end());
        int midpoint = tmp.size() / 2;
        std::sort(tmp.begin(), tmp.end());
        if (tmp.size() % 2 == 0) {
            median = (tmp[midpoint - 1] + tmp[midpoint]) / 2;
        } else {
            median = tmp[midpoint];
        }
        return median;
    }

    float compute_mean(const vector<float> &values) const {
        float sum = 0;
        for (float v : values) {
            sum += v;
        }
        return sum / values.size();
    }

    bool eq(float a, float b) const {
        const float thresh = 0.00001;
        return a == b || (std::abs(a - b) / b) < thresh;
    }

    void tests() const {
        vector<float> values({5, 2, -16, 104, 22});
        assert(eq(compute_min(values), -16));
        assert(eq(compute_max(values), 104));
        assert(eq(compute_mean(values), 23.4));
        assert(eq(compute_median(values), 5));
        values.push_back(12);
        assert(eq(compute_median(values), 8.5));
    }
};

#endif
