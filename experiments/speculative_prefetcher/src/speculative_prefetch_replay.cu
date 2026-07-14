// Real O_DIRECT + io_uring + pinned-memory + CUDA H2D replay for speculative
// selected-block prefetching. This is an isolated microbenchmark, not a
// llama.cpp integration.

#include <cuda_runtime.h>
#include <liburing.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Args {
    std::string file_path;
    std::string events_path;
    std::string output_path;
    std::uint64_t file_size = 2ull * 1024ull * 1024ull * 1024ull;
    std::uint64_t block_bytes = 65536;
    int queue_depth = 4;
    int slot_count = 64;
    std::uint64_t seed = 20260714;
};

struct EventRow {
    std::string event_id;
    std::string mode;
    std::string dataset;
    std::string sample_id;
    int decode_step = 0;
    int layer_id = 0;
    int head_id = 0;
    std::string policy;
    double lead_ms = 0.0;
    double slack_ms = 0.0;
    int candidate_count = 0;
    std::set<int> selected;
    std::set<int> predicted;
};

struct Request {
    int block_id = -1;
    bool correction = false;
};

struct Slot {
    void* host = nullptr;
    int fixed_index = -1;
    bool busy = false;
    bool inflight = false;
    bool copying = false;
    int block_id = -1;
    bool correction = false;
    double submit_ms = 0.0;
    double cqe_ms = 0.0;
    double h2d_enqueue_ms = 0.0;
    double h2d_done_ms = 0.0;
    cudaEvent_t copy_done = nullptr;
};

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "error: " << msg << "\n";
    std::exit(2);
}

void check_cuda(cudaError_t code, const char* op) {
    if (code != cudaSuccess) {
        die(std::string(op) + " failed: " + cudaGetErrorString(code));
    }
}

template <typename T>
T parse_u64(const std::string& value, const std::string& name) {
    try {
        std::size_t pos = 0;
        unsigned long long parsed = std::stoull(value, &pos, 10);
        if (pos != value.size()) die("invalid integer for " + name + ": " + value);
        return static_cast<T>(parsed);
    } catch (...) {
        die("invalid integer for " + name + ": " + value);
    }
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value for " + key);
            return argv[++i];
        };
        if (key == "--file") args.file_path = value();
        else if (key == "--events") args.events_path = value();
        else if (key == "--output") args.output_path = value();
        else if (key == "--file-size") args.file_size = parse_u64<std::uint64_t>(value(), key);
        else if (key == "--block-bytes") args.block_bytes = parse_u64<std::uint64_t>(value(), key);
        else if (key == "--queue-depth") args.queue_depth = parse_u64<int>(value(), key);
        else if (key == "--slot-count") args.slot_count = parse_u64<int>(value(), key);
        else if (key == "--seed") args.seed = parse_u64<std::uint64_t>(value(), key);
        else die("unknown argument: " + key);
    }
    if (args.file_path.empty()) die("--file is required");
    if (args.events_path.empty()) die("--events is required");
    if (args.output_path.empty()) die("--output is required");
    if (args.block_bytes == 0 || args.block_bytes % 4096 != 0) die("--block-bytes must be 4096 aligned");
    if (args.queue_depth <= 0) die("--queue-depth must be positive");
    if (args.slot_count < args.queue_depth) die("--slot-count must be >= queue-depth");
    return args;
}

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::string cur;
    std::stringstream ss(s);
    while (std::getline(ss, cur, delim)) out.push_back(cur);
    return out;
}

std::set<int> parse_ids(const std::string& s) {
    std::set<int> out;
    if (s.empty()) return out;
    for (const auto& part : split(s, ';')) {
        if (!part.empty()) out.insert(std::stoi(part));
    }
    return out;
}

std::string json_escape(const std::string& s) {
    std::ostringstream out;
    for (char c : s) {
        if (c == '\\') out << "\\\\";
        else if (c == '"') out << "\\\"";
        else if (c == '\n') out << "\\n";
        else out << c;
    }
    return out.str();
}

std::vector<EventRow> read_events(const std::string& path) {
    std::ifstream in(path);
    if (!in) die("failed to open events csv: " + path);
    std::string header;
    if (!std::getline(in, header)) die("empty events csv");
    auto fields = split(header, ',');
    std::map<std::string, int> idx;
    for (int i = 0; i < static_cast<int>(fields.size()); ++i) idx[fields[i]] = i;
    auto get = [&](const std::vector<std::string>& row, const std::string& name) -> std::string {
        auto it = idx.find(name);
        if (it == idx.end() || it->second >= static_cast<int>(row.size())) die("missing csv field: " + name);
        return row[it->second];
    };
    std::vector<EventRow> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto row = split(line, ',');
        EventRow ev;
        ev.event_id = get(row, "event_id");
        ev.mode = get(row, "mode");
        ev.dataset = get(row, "dataset");
        ev.sample_id = get(row, "sample_id");
        ev.decode_step = std::stoi(get(row, "decode_step"));
        ev.layer_id = std::stoi(get(row, "layer_id"));
        ev.head_id = std::stoi(get(row, "head_id"));
        ev.policy = get(row, "policy");
        ev.lead_ms = std::stod(get(row, "lead_ms"));
        ev.slack_ms = std::stod(get(row, "slack_ms"));
        ev.candidate_count = std::stoi(get(row, "candidate_count"));
        ev.selected = parse_ids(get(row, "selected_blocks"));
        ev.predicted = parse_ids(get(row, "predicted_blocks"));
        rows.push_back(std::move(ev));
    }
    return rows;
}

double since_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::vector<std::uint64_t> make_offsets(const Args& args, int candidate_count) {
    std::uint64_t file_blocks = args.file_size / args.block_bytes;
    if (file_blocks < static_cast<std::uint64_t>(candidate_count * 4)) die("file too small for candidate_count");
    std::mt19937_64 rng(args.seed + static_cast<std::uint64_t>(candidate_count) * 17ull);
    std::uniform_int_distribution<std::uint64_t> dist(0, file_blocks - 1);
    std::vector<std::uint64_t> chosen;
    while (chosen.size() < static_cast<std::size_t>(candidate_count)) {
        std::uint64_t id = dist(rng);
        bool duplicate = std::find(chosen.begin(), chosen.end(), id) != chosen.end();
        bool adjacent = false;
        for (auto old : chosen) {
            if (old + 1 == id || id + 1 == old) adjacent = true;
        }
        if (!duplicate && !adjacent) chosen.push_back(id);
    }
    std::vector<std::uint64_t> offsets;
    for (auto id : chosen) offsets.push_back(id * args.block_bytes);
    return offsets;
}

int open_checked(const Args& args) {
    struct stat st {};
    if (stat(args.file_path.c_str(), &st) != 0) die("stat file failed: " + std::string(strerror(errno)));
    if (!S_ISREG(st.st_mode)) die("test file is not regular");
    if (static_cast<std::uint64_t>(st.st_size) < args.file_size) die("test file smaller than configured file size");
    int fd = open(args.file_path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd < 0) die("open O_DIRECT failed: " + std::string(strerror(errno)));
    return fd;
}

struct IoContext {
    Args args;
    int fd = -1;
    io_uring ring {};
    std::vector<Slot> slots;
    void* host_base = nullptr;
    std::uint8_t* device_base = nullptr;
    cudaStream_t copy_stream = nullptr;
};

void setup_io(IoContext& ctx) {
    ctx.fd = open_checked(ctx.args);
    int ret = io_uring_queue_init(static_cast<unsigned>(ctx.args.queue_depth * 2 + 8), &ctx.ring, 0);
    if (ret < 0) die("io_uring_queue_init failed: " + std::string(strerror(-ret)));

    std::size_t total = static_cast<std::size_t>(ctx.args.slot_count) * static_cast<std::size_t>(ctx.args.block_bytes);
    if (posix_memalign(&ctx.host_base, 4096, total) != 0) die("posix_memalign failed");
    std::memset(ctx.host_base, 0, total);
    check_cuda(cudaHostRegister(ctx.host_base, total, cudaHostRegisterDefault), "cudaHostRegister");
    check_cuda(cudaMalloc(&ctx.device_base, total), "cudaMalloc device slots");
    check_cuda(cudaStreamCreateWithFlags(&ctx.copy_stream, cudaStreamNonBlocking), "cudaStreamCreate");

    std::vector<iovec> iovecs(ctx.args.slot_count);
    ctx.slots.resize(ctx.args.slot_count);
    for (int i = 0; i < ctx.args.slot_count; ++i) {
        void* ptr = static_cast<char*>(ctx.host_base) + static_cast<std::uint64_t>(i) * ctx.args.block_bytes;
        iovecs[i].iov_base = ptr;
        iovecs[i].iov_len = ctx.args.block_bytes;
        ctx.slots[i].host = ptr;
        ctx.slots[i].fixed_index = i;
        check_cuda(cudaEventCreateWithFlags(&ctx.slots[i].copy_done, cudaEventDisableTiming), "cudaEventCreate");
    }
    ret = io_uring_register_buffers(&ctx.ring, iovecs.data(), static_cast<unsigned>(iovecs.size()));
    if (ret < 0) die("io_uring_register_buffers failed: " + std::string(strerror(-ret)));
}

void teardown_io(IoContext& ctx) {
    if (ctx.copy_stream) cudaStreamSynchronize(ctx.copy_stream);
    for (auto& slot : ctx.slots) {
        if (slot.copy_done) cudaEventDestroy(slot.copy_done);
    }
    if (ctx.device_base) cudaFree(ctx.device_base);
    if (ctx.host_base) {
        cudaHostUnregister(ctx.host_base);
        free(ctx.host_base);
    }
    io_uring_unregister_buffers(&ctx.ring);
    io_uring_queue_exit(&ctx.ring);
    if (ctx.fd >= 0) close(ctx.fd);
}

struct EventResult {
    double wall_ms = 0.0;
    double ts_ms = 0.0;
    double ta_ms = 0.0;
    double attention_start_ms = 0.0;
    double t_block_ms = 0.0;
    double on_time_recall = 0.0;
    int semantic_tp = 0;
    int semantic_fp = 0;
    int semantic_fn = 0;
    std::uint64_t useful_on_time_bytes = 0;
    std::uint64_t late_useful_bytes = 0;
    std::uint64_t wrong_nvme_bytes = 0;
    std::uint64_t wrong_h2d_bytes = 0;
    std::uint64_t correction_bytes = 0;
    std::uint64_t total_read_bytes = 0;
    int request_count = 0;
    int actual_qd_max = 0;
};

bool required_done(const std::set<int>& selected, const std::map<int, double>& done) {
    for (int b : selected) {
        if (done.find(b) == done.end()) return false;
    }
    return true;
}

EventResult run_event(IoContext& ctx, const EventRow& ev, const std::vector<std::uint64_t>& offsets) {
    for (auto& slot : ctx.slots) {
        slot.busy = false;
        slot.inflight = false;
        slot.copying = false;
        slot.block_id = -1;
        slot.correction = false;
        slot.submit_ms = slot.cqe_ms = slot.h2d_enqueue_ms = slot.h2d_done_ms = 0.0;
    }
    check_cuda(cudaStreamSynchronize(ctx.copy_stream), "pre-event copy stream sync");

    std::set<int> tp_set;
    std::set_intersection(ev.selected.begin(), ev.selected.end(), ev.predicted.begin(), ev.predicted.end(), std::inserter(tp_set, tp_set.begin()));
    std::set<int> wrong_set;
    std::set_difference(ev.predicted.begin(), ev.predicted.end(), ev.selected.begin(), ev.selected.end(), std::inserter(wrong_set, wrong_set.begin()));
    std::set<int> missing_set;
    std::set_difference(ev.selected.begin(), ev.selected.end(), ev.predicted.begin(), ev.predicted.end(), std::inserter(missing_set, missing_set.begin()));

    EventResult res;
    res.semantic_tp = static_cast<int>(tp_set.size());
    res.semantic_fp = static_cast<int>(wrong_set.size());
    res.semantic_fn = static_cast<int>(missing_set.size());
    res.wrong_nvme_bytes = wrong_set.size() * ctx.args.block_bytes;
    res.wrong_h2d_bytes = wrong_set.size() * ctx.args.block_bytes;
    res.correction_bytes = missing_set.size() * ctx.args.block_bytes;
    res.total_read_bytes = (ev.predicted.size() + missing_set.size()) * ctx.args.block_bytes;
    res.request_count = static_cast<int>(ev.predicted.size() + missing_set.size());

    std::deque<Request> prefetch_q;
    for (int b : ev.predicted) prefetch_q.push_back({b, false});
    std::deque<Request> correction_q;
    for (int b : missing_set) correction_q.push_back({b, true});

    std::map<int, double> h2d_done;
    std::map<int, bool> done_from_prefetch;
    int inflight_reads = 0;
    int next_slot = 0;
    bool correction_open = false;
    auto start = Clock::now();
    const double ts_deadline = ev.lead_ms;
    const double ta_deadline = ev.lead_ms + ev.slack_ms;

    auto submit_more = [&]() {
        bool submitted_any = false;
        while (inflight_reads < ctx.args.queue_depth) {
            Request req;
            bool has = false;
            if (correction_open && !correction_q.empty()) {
                req = correction_q.front();
                correction_q.pop_front();
                has = true;
            } else if (!prefetch_q.empty()) {
                req = prefetch_q.front();
                prefetch_q.pop_front();
                has = true;
            }
            if (!has) break;
            if (req.block_id < 0 || req.block_id >= static_cast<int>(offsets.size())) die("block id exceeds offset table");
            if (next_slot >= static_cast<int>(ctx.slots.size())) die("not enough fixed buffer slots for event");
            io_uring_sqe* sqe = io_uring_get_sqe(&ctx.ring);
            if (!sqe) break;
            Slot& slot = ctx.slots[next_slot++];
            slot.busy = true;
            slot.inflight = true;
            slot.copying = false;
            slot.block_id = req.block_id;
            slot.correction = req.correction;
            slot.submit_ms = since_ms(start);
            io_uring_prep_read_fixed(sqe, ctx.fd, slot.host, ctx.args.block_bytes, offsets[req.block_id], slot.fixed_index);
            io_uring_sqe_set_data64(sqe, static_cast<std::uint64_t>(slot.fixed_index));
            inflight_reads++;
            res.actual_qd_max = std::max(res.actual_qd_max, inflight_reads);
            submitted_any = true;
        }
        if (submitted_any) {
            int ret = io_uring_submit(&ctx.ring);
            if (ret < 0) die("io_uring_submit failed: " + std::string(strerror(-ret)));
        }
    };

    submit_more();

    while (!required_done(ev.selected, h2d_done)) {
        double now_ms = since_ms(start);
        if (!correction_open && now_ms >= ts_deadline) {
            correction_open = true;
        }
        submit_more();

        io_uring_cqe* cqe = nullptr;
        while (io_uring_peek_cqe(&ctx.ring, &cqe) == 0 && cqe != nullptr) {
            int slot_id = static_cast<int>(io_uring_cqe_get_data64(cqe));
            if (slot_id < 0 || slot_id >= static_cast<int>(ctx.slots.size())) die("invalid CQE slot id");
            Slot& slot = ctx.slots[slot_id];
            if (cqe->res < 0) die("O_DIRECT read failed: " + std::string(strerror(-cqe->res)));
            if (static_cast<std::uint64_t>(cqe->res) != ctx.args.block_bytes) die("short O_DIRECT read");
            slot.cqe_ms = since_ms(start);
            slot.inflight = false;
            inflight_reads--;
            std::uint8_t* dst = ctx.device_base + static_cast<std::uint64_t>(slot.fixed_index) * ctx.args.block_bytes;
            slot.h2d_enqueue_ms = since_ms(start);
            check_cuda(cudaMemcpyAsync(dst, slot.host, ctx.args.block_bytes, cudaMemcpyHostToDevice, ctx.copy_stream), "cudaMemcpyAsync H2D");
            check_cuda(cudaEventRecord(slot.copy_done, ctx.copy_stream), "cudaEventRecord");
            slot.copying = true;
            io_uring_cqe_seen(&ctx.ring, cqe);
            cqe = nullptr;
        }

        for (auto& slot : ctx.slots) {
            if (!slot.copying) continue;
            cudaError_t q = cudaEventQuery(slot.copy_done);
            if (q == cudaSuccess) {
                slot.copying = false;
                slot.h2d_done_ms = since_ms(start);
                h2d_done[slot.block_id] = slot.h2d_done_ms;
                done_from_prefetch[slot.block_id] = !slot.correction;
            } else if (q != cudaErrorNotReady) {
                check_cuda(q, "cudaEventQuery");
            }
        }
    }

    res.attention_start_ms = since_ms(start);
    res.ts_ms = ts_deadline;
    res.ta_ms = ta_deadline;
    res.wall_ms = res.attention_start_ms;
    res.t_block_ms = std::max(0.0, res.attention_start_ms - ta_deadline);

    int on_time_hits = 0;
    int late_hits = 0;
    for (int b : tp_set) {
        auto it = h2d_done.find(b);
        if (it != h2d_done.end() && done_from_prefetch[b] && it->second <= ta_deadline) on_time_hits++;
        else late_hits++;
    }
    res.on_time_recall = ev.selected.empty() ? 1.0 : static_cast<double>(on_time_hits) / static_cast<double>(ev.selected.size());
    res.useful_on_time_bytes = static_cast<std::uint64_t>(on_time_hits) * ctx.args.block_bytes;
    res.late_useful_bytes = static_cast<std::uint64_t>(late_hits) * ctx.args.block_bytes;
    return res;
}

void append_json(std::ofstream& out, const Args& args, const EventRow& ev, const EventResult& r) {
    double semantic_recall = ev.selected.empty() ? 1.0 : static_cast<double>(r.semantic_tp) / static_cast<double>(ev.selected.size());
    double semantic_precision = ev.predicted.empty() ? (ev.selected.empty() ? 1.0 : 0.0) : static_cast<double>(r.semantic_tp) / static_cast<double>(ev.predicted.size());
    double read_amp = ev.selected.empty() ? 0.0 : static_cast<double>(r.total_read_bytes) / static_cast<double>(ev.selected.size() * args.block_bytes);
    out << "{"
        << "\"event_id\":\"" << json_escape(ev.event_id) << "\","
        << "\"mode\":\"" << json_escape(ev.mode) << "\","
        << "\"dataset\":\"" << json_escape(ev.dataset) << "\","
        << "\"sample_id\":\"" << json_escape(ev.sample_id) << "\","
        << "\"decode_step\":" << ev.decode_step << ","
        << "\"layer_id\":" << ev.layer_id << ","
        << "\"head_id\":" << ev.head_id << ","
        << "\"policy\":\"" << json_escape(ev.policy) << "\","
        << "\"queue_depth\":" << args.queue_depth << ","
        << "\"block_bytes\":" << args.block_bytes << ","
        << "\"selected_blocks\":" << ev.selected.size() << ","
        << "\"predicted_blocks\":" << ev.predicted.size() << ","
        << "\"semantic_tp\":" << r.semantic_tp << ","
        << "\"semantic_fp\":" << r.semantic_fp << ","
        << "\"semantic_fn\":" << r.semantic_fn << ","
        << "\"semantic_recall\":" << semantic_recall << ","
        << "\"semantic_precision\":" << semantic_precision << ","
        << "\"on_time_recall\":" << r.on_time_recall << ","
        << "\"prediction_lead_ms\":" << ev.lead_ms << ","
        << "\"correction_slack_ms\":" << ev.slack_ms << ","
        << "\"prefetch_lead_ms\":" << (ev.lead_ms + ev.slack_ms) << ","
        << "\"ts_ms\":" << r.ts_ms << ","
        << "\"ta_ms\":" << r.ta_ms << ","
        << "\"attention_start_ms\":" << r.attention_start_ms << ","
        << "\"t_block_ms\":" << r.t_block_ms << ","
        << "\"useful_on_time_bytes\":" << r.useful_on_time_bytes << ","
        << "\"late_useful_bytes\":" << r.late_useful_bytes << ","
        << "\"wrong_nvme_bytes\":" << r.wrong_nvme_bytes << ","
        << "\"wrong_h2d_bytes\":" << r.wrong_h2d_bytes << ","
        << "\"correction_bytes\":" << r.correction_bytes << ","
        << "\"total_read_bytes\":" << r.total_read_bytes << ","
        << "\"read_amplification\":" << read_amp << ","
        << "\"request_count\":" << r.request_count << ","
        << "\"actual_qd_max\":" << r.actual_qd_max
        << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    check_cuda(cudaFree(0), "cuda init");

    auto events = read_events(args.events_path);
    if (events.empty()) die("no replay events");
    int max_candidate = 0;
    for (const auto& ev : events) max_candidate = std::max(max_candidate, ev.candidate_count);
    auto offsets = make_offsets(args, max_candidate);

    IoContext ctx;
    ctx.args = args;
    setup_io(ctx);

    std::ofstream out(args.output_path);
    if (!out) die("failed to open output: " + args.output_path);
    for (const auto& ev : events) {
        EventResult r = run_event(ctx, ev, offsets);
        append_json(out, args, ev, r);
    }
    teardown_io(ctx);
    return 0;
}
