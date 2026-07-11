#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(HAVE_LIBURING)
#include <liburing.h>
#endif

struct probe_config {
    const char *path;
    size_t block_size;
    size_t block_count;
};

static uint64_t now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void fill_block(unsigned char *buf, size_t size, size_t block_id) {
    for (size_t i = 0; i < size; ++i) {
        buf[i] = (unsigned char)((block_id * 131u + i) & 0xffu);
    }
}

static int verify_block(const unsigned char *buf, size_t size, size_t block_id) {
    for (size_t i = 0; i < size; ++i) {
        unsigned char expected = (unsigned char)((block_id * 131u + i) & 0xffu);
        if (buf[i] != expected) {
            return -1;
        }
    }
    return 0;
}

static int parse_size(const char *text, size_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return -1;
    }
    *out = (size_t)value;
    return 0;
}

#if !defined(HAVE_LIBURING)
static int run_posix_probe(const struct probe_config *cfg, uint64_t *write_ns, uint64_t *read_ns) {
    int fd = open(cfg->path, O_CREAT | O_TRUNC | O_RDWR, 0600);
    if (fd < 0) {
        perror("open");
        return -1;
    }

    unsigned char *buf = NULL;
    if (posix_memalign((void **)&buf, 4096, cfg->block_size) != 0) {
        close(fd);
        return -1;
    }

    uint64_t start = now_ns();
    for (size_t block = 0; block < cfg->block_count; ++block) {
        fill_block(buf, cfg->block_size, block);
        off_t offset = (off_t)(block * cfg->block_size);
        ssize_t written = pwrite(fd, buf, cfg->block_size, offset);
        if (written != (ssize_t)cfg->block_size) {
            perror("pwrite");
            free(buf);
            close(fd);
            return -1;
        }
    }
    if (fsync(fd) != 0) {
        perror("fsync");
        free(buf);
        close(fd);
        return -1;
    }
    *write_ns = now_ns() - start;

    start = now_ns();
    for (size_t block = 0; block < cfg->block_count; ++block) {
        memset(buf, 0, cfg->block_size);
        off_t offset = (off_t)(block * cfg->block_size);
        ssize_t read_bytes = pread(fd, buf, cfg->block_size, offset);
        if (read_bytes != (ssize_t)cfg->block_size) {
            perror("pread");
            free(buf);
            close(fd);
            return -1;
        }
        if (verify_block(buf, cfg->block_size, block) != 0) {
            fprintf(stderr, "verification failed for block %zu\n", block);
            free(buf);
            close(fd);
            return -1;
        }
    }
    *read_ns = now_ns() - start;

    free(buf);
    close(fd);
    return 0;
}

#endif

#if defined(HAVE_LIBURING)
static int run_uring_probe(const struct probe_config *cfg, uint64_t *write_ns, uint64_t *read_ns) {
    /* Keep the first liburing backend intentionally small and synchronous-at-the
       completion boundary. Later phases can add queue depth experiments. */
    int fd = open(cfg->path, O_CREAT | O_TRUNC | O_RDWR, 0600);
    if (fd < 0) {
        perror("open");
        return -1;
    }

    struct io_uring ring;
    if (io_uring_queue_init(8, &ring, 0) != 0) {
        close(fd);
        return -1;
    }

    unsigned char *buf = NULL;
    if (posix_memalign((void **)&buf, 4096, cfg->block_size) != 0) {
        io_uring_queue_exit(&ring);
        close(fd);
        return -1;
    }

    uint64_t start = now_ns();
    for (size_t block = 0; block < cfg->block_count; ++block) {
        fill_block(buf, cfg->block_size, block);
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        io_uring_prep_write(sqe, fd, buf, cfg->block_size, (off_t)(block * cfg->block_size));
        io_uring_submit(&ring);
        struct io_uring_cqe *cqe = NULL;
        int ret = io_uring_wait_cqe(&ring, &cqe);
        if (ret < 0 || cqe->res != (int)cfg->block_size) {
            fprintf(stderr, "io_uring write failed\n");
            free(buf);
            io_uring_queue_exit(&ring);
            close(fd);
            return -1;
        }
        io_uring_cqe_seen(&ring, cqe);
    }
    fsync(fd);
    *write_ns = now_ns() - start;

    start = now_ns();
    for (size_t block = 0; block < cfg->block_count; ++block) {
        memset(buf, 0, cfg->block_size);
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fd, buf, cfg->block_size, (off_t)(block * cfg->block_size));
        io_uring_submit(&ring);
        struct io_uring_cqe *cqe = NULL;
        int ret = io_uring_wait_cqe(&ring, &cqe);
        if (ret < 0 || cqe->res != (int)cfg->block_size) {
            fprintf(stderr, "io_uring read failed\n");
            free(buf);
            io_uring_queue_exit(&ring);
            close(fd);
            return -1;
        }
        io_uring_cqe_seen(&ring, cqe);
        if (verify_block(buf, cfg->block_size, block) != 0) {
            fprintf(stderr, "verification failed for block %zu\n", block);
            free(buf);
            io_uring_queue_exit(&ring);
            close(fd);
            return -1;
        }
    }
    *read_ns = now_ns() - start;

    free(buf);
    io_uring_queue_exit(&ring);
    close(fd);
    return 0;
}
#endif

int main(int argc, char **argv) {
    struct probe_config cfg = {
        .path = "outputs/io_probe.bin",
        .block_size = 4096,
        .block_count = 8,
    };

    if (argc > 1) {
        cfg.path = argv[1];
    }
    if (argc > 2 && parse_size(argv[2], &cfg.block_size) != 0) {
        fprintf(stderr, "invalid block_size: %s\n", argv[2]);
        return 2;
    }
    if (argc > 3 && parse_size(argv[3], &cfg.block_count) != 0) {
        fprintf(stderr, "invalid block_count: %s\n", argv[3]);
        return 2;
    }

    uint64_t write_ns = 0;
    uint64_t read_ns = 0;
    int rc = 0;
#if defined(HAVE_LIBURING)
    const char *backend = "liburing";
    rc = run_uring_probe(&cfg, &write_ns, &read_ns);
#else
    const char *backend = "posix_fallback";
    rc = run_posix_probe(&cfg, &write_ns, &read_ns);
#endif
    if (rc != 0) {
        return 1;
    }

    printf("{\"backend\":\"%s\",\"path\":\"%s\",\"block_size\":%zu,\"block_count\":%zu,\"bytes\":%zu,\"write_latency_ns\":%" PRIu64 ",\"read_latency_ns\":%" PRIu64 "}\n",
           backend,
           cfg.path,
           cfg.block_size,
           cfg.block_count,
           cfg.block_size * cfg.block_count,
           write_ns,
           read_ns);
    return 0;
}
