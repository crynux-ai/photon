#include "include/profiler.h"

#include <cassert>
#include <chrono>
#include <map>
#include <iostream>

int64_t Profiler::now_ns() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
}
void Profiler::begin(int obj_id, const std::string& tag) {
    add(obj_id, tag, now_ns(), -1);
}

void Profiler::end(int obj_id, const std::string& tag) {
    assert(!metrics[obj_id][tag].empty());
    for (auto& item : metrics[obj_id][tag]) {
        if (item.end_ns < 0) {
            item.end_ns = now_ns();
            break;
        }
    }
}

void Profiler::add(int obj_id, const std::string& tag, int64_t begin_ns, int64_t end_ns) {
    if (metrics.find(obj_id) == metrics.end()) {
        metrics[obj_id] = {};
    }
    auto& item = metrics[obj_id];
    if (item.find(tag) == item.end()) {
        item[tag] = {};
    }
    item[tag].push_back(Metric{
        obj_id,
        tag,
        begin_ns,
        end_ns,
        /*cnt=*/1,
    });
}

void Profiler::print() {
    std::cout << "Profile metrics: " << std::endl;
    std::map<std::string, uint64_t> duration;
    for (const auto& objs : metrics) {
        for (const auto& item : objs.second) {
            if (duration.find(item.first) == duration.end()) {
                duration[item.first] = 0;
            }
            for (const auto& metric : item.second) {
                assert(metric.end_ns > 0);
                duration[item.first] += metric.end_ns - metric.begin_ns;
            }
        }
    }

    std::vector<std::pair<std::string, uint64_t>> sorted_duration;
    for (const auto& pair : duration) {
        sorted_duration.push_back(pair);
    }
    std::sort(sorted_duration.begin(), sorted_duration.end(),
        [](const std::pair<std::string, uint64_t>& a, const std::pair<std::string, uint64_t>& b) {
            return a.second > b.second;
        });

    for (auto item : sorted_duration) {
        int cnt = item.first.length() / 8;
        cnt = cnt < 5 ? 5 - cnt : 1;
        std::string tab;
        for (int i = 0; i < cnt; i++) {
            tab += "\t";
        }
        std::cout << "\t" << item.first << tab << item.second << std::endl;
    }
}

void Profiler::clear() {
    metrics.clear();
}


std::map<int, std::map<std::string, std::vector<Metric>>> Profiler::metrics;

#ifdef ENABLE_PROFILE
    #define PROFILE_BEGIN(obj_id, tag) Profiler::begin(obj_id, tag);
    #define PROFILE_END(obj_id, tag) Profiler::end(obj_id, tag);
    #define PROFILE_ADD(obj_id, tag, begin, end) Profiler::add(obj_id, tag, begin, end);
    #define PROFILE_PRINT Profiler::print(); Profiler::clear();
#else
    #define PROFILE_BEGIN(obj_id, tag)
    #define PROFILE_END(obj_id, tag)
    #define PROFILE_ADD(obj_id, tag, begin, end)
    #define PROFILE_PRINT
#endif

