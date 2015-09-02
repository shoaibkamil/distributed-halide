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
        _usempi = true;
        _reduced = -1.0;
        _percentile_lower = -1.0;
        _percentile_upper = -1.0;
        _gathered = -1.0;
        _gathered_percentile_lower = -1.0;
        _gathered_percentile_upper = -1.0;
    }

    MPITiming() {
        _usempi = false;
        _reduced = -1.0;
        _percentile_lower = -1.0;
        _percentile_upper = -1.0;
        _gathered = -1.0;
        _gathered_percentile_lower = -1.0;
        _gathered_percentile_upper = -1.0;
    }

    inline void barrier() const {
        MPI_Barrier(_comm);
    }

    void report() const {
        if (_rank == 0) {
            printf("Timing: <%d> ranks <%f> seconds, 20/80 percentile <%f,%f>\n",
                   _numprocs, gathered(), _gathered_percentile_lower, _gathered_percentile_upper);
        }
    }

    void nondistributed_report() const {
        if (_rank == 0) {
            printf("Timing: non-distributed <%f> seconds, 20/80 percentile <%f,%f>\n",
                   _reduced, _percentile_lower, _percentile_upper);
        }
    }

    void record(float sec) {
        _timings.push_back(sec);
    }

    // Reduce all recorded values by the given statistic.
    void reduce(Stat stat) {
        _percentile_lower = compute_percentile(_timings, 20);
        _percentile_upper = compute_percentile(_timings, 80);
        _reduced = compute_stat(_timings, stat);
        _timings.clear();
    }

    // Gather reduced values from all ranks to rank 0, and reduce the
    // gathered values by the given statistic.
    void gather(Stat stat) {
        assert(_usempi);
        const int tag = 0;
        if (_rank == 0) {
            vector<float> results;
            vector<float> percentilelower, percentileupper;
            results.push_back(_reduced);
            percentilelower.push_back(_percentile_lower);
            percentileupper.push_back(_percentile_upper);
            for (int i = 1; i < _numprocs; i++) {
                float t;
                MPI_Recv(&t, 1, MPI_FLOAT, i, tag, _comm, MPI_STATUS_IGNORE);
                results.push_back(t);
                MPI_Recv(&t, 1, MPI_FLOAT, i, tag, _comm, MPI_STATUS_IGNORE);
                percentilelower.push_back(t);
                MPI_Recv(&t, 1, MPI_FLOAT, i, tag, _comm, MPI_STATUS_IGNORE);
                percentileupper.push_back(t);
            }
            _gathered = compute_stat(results, stat);
            _gathered_percentile_lower = compute_stat(percentilelower, Min);
            _gathered_percentile_upper = compute_stat(percentileupper, Max);
        } else {
            MPI_Send(&_reduced, 1, MPI_FLOAT, 0, tag, _comm);
            MPI_Send(&_percentile_lower, 1, MPI_FLOAT, 0, tag, _comm);
            MPI_Send(&_percentile_upper, 1, MPI_FLOAT, 0, tag, _comm);
        }
    }

    // Return the gathered result.
    float gathered() const {
        assert(_usempi);
        return _gathered;
    }
private:
    bool _usempi;
    int _rank, _numprocs;
    MPI_Comm _comm;
    vector<float> _timings;
    float _reduced;
    float _percentile_lower, _percentile_upper;
    float _gathered;
    float _gathered_percentile_lower, _gathered_percentile_upper;

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

    float compute_percentile(const vector<float> &values, int percentile) const {
        assert(0 < percentile && percentile <= 100);
        if (values.size() == 1) return values[0];
        float result = 0;
        vector<float> tmp(values.begin(), values.end());
        std::sort(tmp.begin(), tmp.end());
        float left = 0, right = 0;
        unsigned k = 0;
        for (unsigned i = 0; i < tmp.size(); i++) {
            float rank = (100.0f/tmp.size()) * (i + 0.5);
            if (percentile > rank) {
                left = rank;
            } else if (percentile < rank) {
                right = rank;
                k = i;
                break;
            } else {
                return tmp[i];
            }
        }
        if (k == tmp.size() - 1) {
            return tmp[k];
        }
        result = tmp[k] + tmp.size() * ((percentile - left) / (right - left)) * (tmp[k+1] - tmp[k]);
        return result;
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
