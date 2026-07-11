CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
TARGETS := block_io_probe runtime_kv_command_consumer
BACKEND_MODE ?= auto
LIBURING_CFLAGS ?=
LIBURING_LDFLAGS ?=

ifneq ($(words $(BACKEND_MODE)),1)
  $(error BACKEND_MODE must be auto, posix, or liburing)
endif
ifeq ($(filter auto posix liburing,$(BACKEND_MODE)),)
  $(error BACKEND_MODE must be auto, posix, or liburing)
endif

ifeq ($(BACKEND_MODE),posix)
  LIBURING_AVAILABLE := 0
else
  LIBURING_AVAILABLE := $(shell tmp=$$(mktemp "$${TMPDIR:-/tmp}/solid_attention_liburing_check.XXXXXX") && trap 'rm -f "$$tmp"' 0 1 2 15 && printf '#include <liburing.h>\nint main(void){return 0;}\n' | $(CC) $(LIBURING_CFLAGS) -x c - $(LIBURING_LDFLAGS) -luring -o "$$tmp" >/dev/null 2>&1 && echo 1 || echo 0)
endif

ifeq ($(BACKEND_MODE),liburing)
  ifneq ($(LIBURING_AVAILABLE),1)
    $(error BACKEND_MODE=liburing requested but liburing is unavailable)
  endif
endif

BACKEND_CFLAGS :=
BACKEND_LDFLAGS :=
BACKEND_LDLIBS :=
ifeq ($(LIBURING_AVAILABLE),1)
  BACKEND_CFLAGS := -DHAVE_LIBURING $(LIBURING_CFLAGS)
  BACKEND_LDFLAGS := $(LIBURING_LDFLAGS)
  BACKEND_LDLIBS := -luring
  BACKEND := liburing
else
  BACKEND := posix_fallback
endif
BACKEND_STAMP := ../build/io/backend-$(BACKEND)

.PHONY: all clean smoke backend

all: $(TARGETS)

$(BACKEND_STAMP):
	@mkdir -p $(dir $@)
	@rm -f ../build/io/backend-*
	@touch $@

block_io_probe: block_io_probe.c $(BACKEND_STAMP)
	$(CC) $(CFLAGS) $(BACKEND_CFLAGS) -o $@ $< $(LDFLAGS) $(BACKEND_LDFLAGS) $(LDLIBS) $(BACKEND_LDLIBS)

runtime_kv_command_consumer: runtime_kv_command_consumer.c $(BACKEND_STAMP)
	$(CC) $(CFLAGS) $(BACKEND_CFLAGS) -o $@ $< $(LDFLAGS) $(BACKEND_LDFLAGS) $(LDLIBS) $(BACKEND_LDLIBS)

backend:
	@echo $(BACKEND)

smoke: block_io_probe
	mkdir -p ../outputs/phase2
	./block_io_probe ../outputs/phase2/io_probe.bin 4096 4 > ../outputs/phase2/io_probe_metrics.json
	cat ../outputs/phase2/io_probe_metrics.json

clean:
	rm -f $(TARGETS) ../build/io/backend-*
