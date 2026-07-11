#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(HAVE_LIBURING)
#include <liburing.h>
#endif

#define LOCATION_DRAM 1
#define LOCATION_VRAM 2

struct kv_command {
    char op[32];
    int layer_id;
    int block_id;
    long long start_token;
    long long end_token;
    size_t size_bytes;
    char source[16];
    char target[16];
    long long ssd_offset;
    bool blocking;
};

struct buffer_slot {
    int layer_id;
    int block_id;
    long long start_token;
    long long end_token;
    size_t size_bytes;
    unsigned char *dram;
    unsigned char *vram;
};

struct command_vec {
    struct kv_command *items;
    size_t count;
    size_t capacity;
};

struct slot_vec {
    struct buffer_slot *items;
    size_t count;
    size_t capacity;
};

struct run_stats {
    size_t command_count;
    size_t kv_load_count;
    size_t kv_prefetch_count;
    size_t kv_compute_count;
    size_t kv_evict_count;
    size_t ssd_read_count;
    size_t ssd_write_count;
    size_t dram_to_vram_copy_count;
    size_t vram_compute_count;
    size_t lazy_init_count;
    size_t ssd_seed_count;
    size_t ssd_read_bytes;
    size_t ssd_write_bytes;
    size_t dram_to_vram_bytes;
    size_t lazy_init_bytes;
    size_t ssd_seed_bytes;
    size_t verification_errors;
    uint64_t ssd_read_ns;
    uint64_t ssd_write_ns;
    uint64_t copy_ns;
    uint64_t compute_verify_ns;
};

static uint64_t now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static unsigned char pattern_byte(const struct kv_command *cmd, size_t i) {
    uint64_t seed = 1469598103934665603ull;
    seed ^= (uint64_t)(cmd->layer_id + 1) * 1099511628211ull;
    seed ^= (uint64_t)(cmd->block_id + 17) * 14029467366897019727ull;
    seed ^= (uint64_t)(cmd->start_token + 31) * 1609587929392839161ull;
    seed ^= (uint64_t)(cmd->end_token + 47) * 9650029242287828579ull;
    seed ^= (uint64_t)(cmd->ssd_offset + 59) * 2870177450012600261ull;
    return (unsigned char)((seed + (uint64_t)i * 131u + ((uint64_t)i >> 3)) & 0xffu);
}

static void fill_pattern(unsigned char *buf, size_t size, const struct kv_command *cmd) {
    for (size_t i = 0; i < size; ++i) {
        buf[i] = pattern_byte(cmd, i);
    }
}

static int verify_pattern(const unsigned char *buf, size_t size, const struct kv_command *cmd) {
    for (size_t i = 0; i < size; ++i) {
        if (buf[i] != pattern_byte(cmd, i)) {
            return -1;
        }
    }
    return 0;
}

static int read_entire_line(FILE *fp, char **line, size_t *cap) {
    if (*line == NULL || *cap == 0) {
        *cap = 4096;
        *line = (char *)malloc(*cap);
        if (*line == NULL) {
            return -1;
        }
    }

    size_t len = 0;
    int ch = 0;
    while ((ch = fgetc(fp)) != EOF) {
        if (len + 2 >= *cap) {
            size_t next = *cap * 2;
            char *tmp = (char *)realloc(*line, next);
            if (tmp == NULL) {
                return -1;
            }
            *line = tmp;
            *cap = next;
        }
        (*line)[len++] = (char)ch;
        if (ch == '\n') {
            break;
        }
    }
    if (len == 0 && ch == EOF) {
        return 0;
    }
    (*line)[len] = '\0';
    return 1;
}

static const char *field_after_colon(const char *line, const char *field) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", field);
    const char *pos = strstr(line, needle);
    if (pos == NULL) {
        return NULL;
    }
    pos += strlen(needle);
    while (*pos == ' ' || *pos == '\t') {
        ++pos;
    }
    if (*pos != ':') {
        return NULL;
    }
    ++pos;
    while (*pos == ' ' || *pos == '\t') {
        ++pos;
    }
    return pos;
}

static int parse_string_field(const char *line, const char *field, char *out, size_t out_size) {
    const char *pos = field_after_colon(line, field);
    if (pos == NULL || *pos != '"') {
        return -1;
    }
    ++pos;
    size_t len = 0;
    while (pos[len] != '\0' && pos[len] != '"') {
        if (pos[len] == '\\') {
            return -1;
        }
        ++len;
    }
    if (pos[len] != '"' || len + 1 > out_size) {
        return -1;
    }
    memcpy(out, pos, len);
    out[len] = '\0';
    return 0;
}

static int parse_ll_field(const char *line, const char *field, long long *out) {
    const char *pos = field_after_colon(line, field);
    if (pos == NULL) {
        return -1;
    }
    char *end = NULL;
    errno = 0;
    long long value = strtoll(pos, &end, 10);
    if (errno != 0 || end == pos) {
        return -1;
    }
    *out = value;
    return 0;
}

static int parse_size_field(const char *line, const char *field, size_t *out) {
    long long value = 0;
    if (parse_ll_field(line, field, &value) != 0 || value <= 0) {
        return -1;
    }
    *out = (size_t)value;
    return 0;
}

static int parse_bool_field(const char *line, const char *field, bool *out) {
    const char *pos = field_after_colon(line, field);
    if (pos == NULL) {
        return -1;
    }
    if (strncmp(pos, "true", 4) == 0) {
        *out = true;
        return 0;
    }
    if (strncmp(pos, "false", 5) == 0) {
        *out = false;
        return 0;
    }
    return -1;
}

static int command_vec_push(struct command_vec *vec, const struct kv_command *cmd) {
    if (vec->count == vec->capacity) {
        size_t next = vec->capacity == 0 ? 16 : vec->capacity * 2;
        struct kv_command *items = (struct kv_command *)realloc(vec->items, next * sizeof(struct kv_command));
        if (items == NULL) {
            return -1;
        }
        vec->items = items;
        vec->capacity = next;
    }
    vec->items[vec->count++] = *cmd;
    return 0;
}

static int parse_command_line(const char *line, struct kv_command *cmd) {
    memset(cmd, 0, sizeof(*cmd));
    long long layer = 0;
    long long block = 0;
    if (parse_string_field(line, "op", cmd->op, sizeof(cmd->op)) != 0 ||
            parse_ll_field(line, "layer_id", &layer) != 0 ||
            parse_ll_field(line, "block_id", &block) != 0 ||
            parse_ll_field(line, "start_token", &cmd->start_token) != 0 ||
            parse_ll_field(line, "end_token", &cmd->end_token) != 0 ||
            parse_size_field(line, "size_bytes", &cmd->size_bytes) != 0 ||
            parse_string_field(line, "source", cmd->source, sizeof(cmd->source)) != 0 ||
            parse_string_field(line, "target", cmd->target, sizeof(cmd->target)) != 0 ||
            parse_ll_field(line, "ssd_offset", &cmd->ssd_offset) != 0 ||
            parse_bool_field(line, "blocking", &cmd->blocking) != 0) {
        return -1;
    }
    if (layer < 0 || block < 0 || cmd->start_token < 0 || cmd->end_token <= cmd->start_token || cmd->ssd_offset < 0) {
        return -1;
    }
    cmd->layer_id = (int)layer;
    cmd->block_id = (int)block;
    return 0;
}

static int load_commands(const char *path, struct command_vec *commands) {
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        perror("fopen commands");
        return -1;
    }

    char *line = NULL;
    size_t cap = 0;
    int status = 0;
    while ((status = read_entire_line(fp, &line, &cap)) > 0) {
        if (line[0] == '\0' || line[0] == '\n') {
            continue;
        }
        struct kv_command cmd;
        if (parse_command_line(line, &cmd) != 0 || command_vec_push(commands, &cmd) != 0) {
            fprintf(stderr, "invalid command line: %s", line);
            free(line);
            fclose(fp);
            return -1;
        }
    }
    free(line);
    fclose(fp);
    return status < 0 ? -1 : 0;
}

static int slot_vec_push(struct slot_vec *vec, const struct buffer_slot *slot) {
    if (vec->count == vec->capacity) {
        size_t next = vec->capacity == 0 ? 16 : vec->capacity * 2;
        struct buffer_slot *items = (struct buffer_slot *)realloc(vec->items, next * sizeof(struct buffer_slot));
        if (items == NULL) {
            return -1;
        }
        vec->items = items;
        vec->capacity = next;
    }
    vec->items[vec->count++] = *slot;
    return 0;
}

static struct buffer_slot *get_slot(struct slot_vec *slots, const struct kv_command *cmd) {
    for (size_t i = 0; i < slots->count; ++i) {
        struct buffer_slot *slot = &slots->items[i];
        if (slot->layer_id == cmd->layer_id && slot->block_id == cmd->block_id) {
            return slot;
        }
    }
    struct buffer_slot slot;
    memset(&slot, 0, sizeof(slot));
    slot.layer_id = cmd->layer_id;
    slot.block_id = cmd->block_id;
    slot.start_token = cmd->start_token;
    slot.end_token = cmd->end_token;
    slot.size_bytes = cmd->size_bytes;
    if (slot_vec_push(slots, &slot) != 0) {
        return NULL;
    }
    return &slots->items[slots->count - 1];
}

static unsigned char **slot_buffer_ptr(struct buffer_slot *slot, int location) {
    if (location == LOCATION_DRAM) {
        return &slot->dram;
    }
    if (location == LOCATION_VRAM) {
        return &slot->vram;
    }
    return NULL;
}

static int location_id(const char *location) {
    if (strcmp(location, "dram") == 0) {
        return LOCATION_DRAM;
    }
    if (strcmp(location, "vram") == 0) {
        return LOCATION_VRAM;
    }
    return 0;
}

static unsigned char *ensure_buffer(struct buffer_slot *slot, int location, const struct kv_command *cmd, struct run_stats *stats) {
    unsigned char **ptr = slot_buffer_ptr(slot, location);
    if (ptr == NULL) {
        return NULL;
    }
    if (*ptr == NULL) {
        *ptr = (unsigned char *)malloc(cmd->size_bytes);
        if (*ptr == NULL) {
            return NULL;
        }
        fill_pattern(*ptr, cmd->size_bytes, cmd);
        stats->lazy_init_count += 1;
        stats->lazy_init_bytes += cmd->size_bytes;
    }
    return *ptr;
}

static unsigned char *alloc_target_buffer(struct buffer_slot *slot, int location, const struct kv_command *cmd) {
    unsigned char **ptr = slot_buffer_ptr(slot, location);
    if (ptr == NULL) {
        return NULL;
    }
    if (*ptr == NULL) {
        *ptr = (unsigned char *)malloc(cmd->size_bytes);
        if (*ptr == NULL) {
            return NULL;
        }
    }
    return *ptr;
}

#if !defined(HAVE_LIBURING)
static int full_pread(int fd, unsigned char *buf, size_t size, off_t offset) {
    size_t done = 0;
    while (done < size) {
        ssize_t rc = pread(fd, buf + done, size - done, offset + (off_t)done);
        if (rc <= 0) {
            return -1;
        }
        done += (size_t)rc;
    }
    return 0;
}

static int full_pwrite(int fd, const unsigned char *buf, size_t size, off_t offset) {
    size_t done = 0;
    while (done < size) {
        ssize_t rc = pwrite(fd, buf + done, size - done, offset + (off_t)done);
        if (rc <= 0) {
            return -1;
        }
        done += (size_t)rc;
    }
    return 0;
}
#endif

#if defined(HAVE_LIBURING)
static int uring_rw(struct io_uring *ring, int fd, unsigned char *buf, size_t size, off_t offset, bool write_op) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
    if (sqe == NULL) {
        return -1;
    }
    if (write_op) {
        io_uring_prep_write(sqe, fd, buf, size, offset);
    } else {
        io_uring_prep_read(sqe, fd, buf, size, offset);
    }
    if (io_uring_submit(ring) < 0) {
        return -1;
    }
    struct io_uring_cqe *cqe = NULL;
    int ret = io_uring_wait_cqe(ring, &cqe);
    if (ret < 0 || cqe == NULL) {
        return -1;
    }
    int res = cqe->res;
    io_uring_cqe_seen(ring, cqe);
    return res == (int)size ? 0 : -1;
}
#endif

static int ssd_read_block(int fd, void *ring_ptr, unsigned char *buf, const struct kv_command *cmd, struct run_stats *stats) {
    uint64_t start = now_ns();
#if defined(HAVE_LIBURING)
    int rc = uring_rw((struct io_uring *)ring_ptr, fd, buf, cmd->size_bytes, (off_t)cmd->ssd_offset, false);
#else
    (void)ring_ptr;
    int rc = full_pread(fd, buf, cmd->size_bytes, (off_t)cmd->ssd_offset);
#endif
    stats->ssd_read_ns += now_ns() - start;
    if (rc == 0) {
        stats->ssd_read_count += 1;
        stats->ssd_read_bytes += cmd->size_bytes;
    }
    return rc;
}

static int ssd_write_block(int fd, void *ring_ptr, unsigned char *buf, const struct kv_command *cmd, struct run_stats *stats) {
    uint64_t start = now_ns();
#if defined(HAVE_LIBURING)
    int rc = uring_rw((struct io_uring *)ring_ptr, fd, buf, cmd->size_bytes, (off_t)cmd->ssd_offset, true);
#else
    (void)ring_ptr;
    int rc = full_pwrite(fd, buf, cmd->size_bytes, (off_t)cmd->ssd_offset);
#endif
    stats->ssd_write_ns += now_ns() - start;
    if (rc == 0) {
        stats->ssd_write_count += 1;
        stats->ssd_write_bytes += cmd->size_bytes;
    }
    return rc;
}

static int seed_ssd_reads(int fd, void *ring_ptr, const struct command_vec *commands, struct run_stats *stats) {
    for (size_t i = 0; i < commands->count; ++i) {
        const struct kv_command *cmd = &commands->items[i];
        if (strcmp(cmd->source, "ssd") != 0) {
            continue;
        }
        bool seen = false;
        for (size_t j = 0; j < i; ++j) {
            const struct kv_command *prev = &commands->items[j];
            if (strcmp(prev->source, "ssd") == 0 && prev->ssd_offset == cmd->ssd_offset && prev->size_bytes == cmd->size_bytes) {
                seen = true;
                break;
            }
        }
        if (seen) {
            continue;
        }
        unsigned char *buf = (unsigned char *)malloc(cmd->size_bytes);
        if (buf == NULL) {
            return -1;
        }
        fill_pattern(buf, cmd->size_bytes, cmd);
        uint64_t start = now_ns();
#if defined(HAVE_LIBURING)
        int rc = uring_rw((struct io_uring *)ring_ptr, fd, buf, cmd->size_bytes, (off_t)cmd->ssd_offset, true);
#else
        int rc = full_pwrite(fd, buf, cmd->size_bytes, (off_t)cmd->ssd_offset);
#endif
        stats->ssd_write_ns += now_ns() - start;
        free(buf);
        if (rc != 0) {
            return -1;
        }
        stats->ssd_seed_count += 1;
        stats->ssd_seed_bytes += cmd->size_bytes;
    }
    return 0;
}

static long long max_ssd_end(const struct command_vec *commands) {
    long long end = 0;
    for (size_t i = 0; i < commands->count; ++i) {
        const struct kv_command *cmd = &commands->items[i];
        long long current = cmd->ssd_offset + (long long)cmd->size_bytes;
        if (current > end) {
            end = current;
        }
    }
    return end;
}

static int execute_command(int fd, void *ring_ptr, struct slot_vec *slots, const struct kv_command *cmd, struct run_stats *stats) {
    struct buffer_slot *slot = get_slot(slots, cmd);
    if (slot == NULL) {
        return -1;
    }

    if (strcmp(cmd->op, "kv_load") == 0 || strcmp(cmd->op, "kv_prefetch") == 0) {
        if (strcmp(cmd->op, "kv_load") == 0) {
            stats->kv_load_count += 1;
        } else {
            stats->kv_prefetch_count += 1;
        }
        int target = location_id(cmd->target);
        unsigned char *dst = alloc_target_buffer(slot, target, cmd);
        if (dst == NULL) {
            return -1;
        }
        if (strcmp(cmd->source, "ssd") == 0) {
            if (ssd_read_block(fd, ring_ptr, dst, cmd, stats) != 0) {
                return -1;
            }
            if (verify_pattern(dst, cmd->size_bytes, cmd) != 0) {
                stats->verification_errors += 1;
                return -1;
            }
            return 0;
        }
        int source = location_id(cmd->source);
        unsigned char *src = ensure_buffer(slot, source, cmd, stats);
        if (src == NULL) {
            return -1;
        }
        uint64_t start = now_ns();
        memcpy(dst, src, cmd->size_bytes);
        stats->copy_ns += now_ns() - start;
        if (strcmp(cmd->source, "dram") == 0 && strcmp(cmd->target, "vram") == 0) {
            stats->dram_to_vram_copy_count += 1;
            stats->dram_to_vram_bytes += cmd->size_bytes;
        }
        return 0;
    }

    if (strcmp(cmd->op, "kv_compute") == 0) {
        stats->kv_compute_count += 1;
        struct buffer_slot *compute_slot = get_slot(slots, cmd);
        unsigned char *src = ensure_buffer(compute_slot, LOCATION_VRAM, cmd, stats);
        if (src == NULL) {
            return -1;
        }
        uint64_t start = now_ns();
        if (verify_pattern(src, cmd->size_bytes, cmd) != 0) {
            stats->verification_errors += 1;
            return -1;
        }
        stats->compute_verify_ns += now_ns() - start;
        stats->vram_compute_count += 1;
        return 0;
    }

    if (strcmp(cmd->op, "kv_evict") == 0) {
        stats->kv_evict_count += 1;
        int source = location_id(cmd->source);
        unsigned char *src = ensure_buffer(slot, source, cmd, stats);
        if (src == NULL) {
            return -1;
        }
        return ssd_write_block(fd, ring_ptr, src, cmd, stats);
    }

    fprintf(stderr, "unsupported op: %s\n", cmd->op);
    return -1;
}

static void free_slots(struct slot_vec *slots) {
    for (size_t i = 0; i < slots->count; ++i) {
        free(slots->items[i].dram);
        free(slots->items[i].vram);
    }
    free(slots->items);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <commands.jsonl> <ssd_backing_file>\n", argv[0]);
        return 2;
    }

    struct command_vec commands = {0};
    if (load_commands(argv[1], &commands) != 0 || commands.count == 0) {
        free(commands.items);
        return 2;
    }

    int fd = open(argv[2], O_CREAT | O_TRUNC | O_RDWR, 0600);
    if (fd < 0) {
        perror("open backing file");
        free(commands.items);
        return 1;
    }
    long long ssd_size = max_ssd_end(&commands);
    if (ssd_size > 0 && ftruncate(fd, (off_t)ssd_size) != 0) {
        perror("ftruncate");
        close(fd);
        free(commands.items);
        return 1;
    }

    struct run_stats stats = {0};
    stats.command_count = commands.count;

#if defined(HAVE_LIBURING)
    const char *backend = "liburing";
    struct io_uring ring;
    if (io_uring_queue_init(8, &ring, 0) != 0) {
        close(fd);
        free(commands.items);
        return 1;
    }
    void *ring_ptr = &ring;
#else
    const char *backend = "posix_fallback";
    void *ring_ptr = NULL;
#endif

    int rc = 0;
    if (seed_ssd_reads(fd, ring_ptr, &commands, &stats) != 0) {
        rc = 1;
    }

    struct slot_vec slots = {0};
    if (rc == 0) {
        for (size_t i = 0; i < commands.count; ++i) {
            if (execute_command(fd, ring_ptr, &slots, &commands.items[i], &stats) != 0) {
                rc = 1;
                break;
            }
        }
    }

#if defined(HAVE_LIBURING)
    io_uring_queue_exit(&ring);
#endif
    fsync(fd);
    close(fd);
    free_slots(&slots);
    free(commands.items);

    if (rc != 0) {
        return 1;
    }

    printf("{\"backend\":\"%s\",\"command_count\":%zu,\"kv_load_count\":%zu,\"kv_prefetch_count\":%zu,\"kv_compute_count\":%zu,\"kv_evict_count\":%zu,\"ssd_read_count\":%zu,\"ssd_write_count\":%zu,\"dram_to_vram_copy_count\":%zu,\"vram_compute_count\":%zu,\"lazy_init_count\":%zu,\"ssd_seed_count\":%zu,\"ssd_read_bytes\":%zu,\"ssd_write_bytes\":%zu,\"dram_to_vram_bytes\":%zu,\"lazy_init_bytes\":%zu,\"ssd_seed_bytes\":%zu,\"verification_errors\":%zu,\"ssd_read_latency_ns\":%" PRIu64 ",\"ssd_write_latency_ns\":%" PRIu64 ",\"copy_latency_ns\":%" PRIu64 ",\"compute_verify_latency_ns\":%" PRIu64 "}\n",
           backend,
           stats.command_count,
           stats.kv_load_count,
           stats.kv_prefetch_count,
           stats.kv_compute_count,
           stats.kv_evict_count,
           stats.ssd_read_count,
           stats.ssd_write_count,
           stats.dram_to_vram_copy_count,
           stats.vram_compute_count,
           stats.lazy_init_count,
           stats.ssd_seed_count,
           stats.ssd_read_bytes,
           stats.ssd_write_bytes,
           stats.dram_to_vram_bytes,
           stats.lazy_init_bytes,
           stats.ssd_seed_bytes,
           stats.verification_errors,
           stats.ssd_read_ns,
           stats.ssd_write_ns,
           stats.copy_ns,
           stats.compute_verify_ns);
    return 0;
}
