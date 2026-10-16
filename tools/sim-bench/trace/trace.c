// LD_PRELOAD logger: every open/openat/stat/access/readlink/ioctl/socket/connect a process makes,
// with its result and errno, to stderr (or $TRACE_FILE). For finding what a closed driver
// checks before it gives up, where ptrace is not allowed.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

static int out_fd = -1;
static __thread int busy;

static void out(const char *fmt, ...) {
    if (busy) return;
    busy = 1;
    if (out_fd < 0) {
        const char *f = getenv("TRACE_FILE");
        out_fd = 2;
        if (f) {
            int (*real_open)(const char *, int, ...) = dlsym(RTLD_NEXT, "open");
            int fd = real_open ? real_open(f, O_WRONLY | O_CREAT | O_APPEND, 0644) : -1;
            if (fd >= 0) out_fd = fd;
        }
    }
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n > 0) {
        if (n > (int)sizeof buf - 1) n = sizeof buf - 1;
        ssize_t w = write(out_fd, buf, (size_t)n);
        (void)w;
    }
    busy = 0;
}

static const char *fdpath(int fd, char *buf, size_t len) {
    char link[64];
    snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
    ssize_t (*real_readlink)(const char *, char *, size_t) = dlsym(RTLD_NEXT, "readlink");
    ssize_t n = real_readlink ? real_readlink(link, buf, len - 1) : -1;
    if (n < 0) return "?";
    buf[n] = 0;
    return buf;
}

#define RESULT(r) (r), ((r) < 0 ? strerror(errno) : "ok")

#define OPEN_LIKE(name)                                                                  \
    int name(const char *path, int flags, ...) {                                        \
        static int (*real)(const char *, int, ...);                                     \
        if (!real) real = dlsym(RTLD_NEXT, #name);                                      \
        mode_t mode = 0;                                                                \
        if (flags & (O_CREAT | __O_TMPFILE)) {                                          \
            va_list ap;                                                                 \
            va_start(ap, flags);                                                        \
            mode = va_arg(ap, mode_t);                                                  \
            va_end(ap);                                                                 \
        }                                                                               \
        int e = errno;                                                                  \
        int r = real(path, flags, mode);                                                \
        int saved = errno;                                                              \
        out("TRACE " #name "(\"%s\", 0x%x) = %d %s\n", path ? path : "(null)", flags, RESULT(r)); \
        errno = r < 0 ? saved : e;                                                      \
        return r;                                                                       \
    }
OPEN_LIKE(open)
OPEN_LIKE(open64)

#define OPENAT_LIKE(name)                                                                \
    int name(int dirfd, const char *path, int flags, ...) {                             \
        static int (*real)(int, const char *, int, ...);                                \
        if (!real) real = dlsym(RTLD_NEXT, #name);                                      \
        mode_t mode = 0;                                                                \
        if (flags & (O_CREAT | __O_TMPFILE)) {                                          \
            va_list ap;                                                                 \
            va_start(ap, flags);                                                        \
            mode = va_arg(ap, mode_t);                                                  \
            va_end(ap);                                                                 \
        }                                                                               \
        int r = real(dirfd, path, flags, mode);                                         \
        int saved = errno;                                                              \
        out("TRACE " #name "(%d, \"%s\", 0x%x) = %d %s\n", dirfd, path ? path : "(null)", flags, RESULT(r)); \
        errno = saved;                                                                  \
        return r;                                                                       \
    }
OPENAT_LIKE(openat)
OPENAT_LIKE(openat64)

int ioctl(int fd, unsigned long req, ...) {
    static int (*real)(int, unsigned long, ...);
    if (!real) real = dlsym(RTLD_NEXT, "ioctl");
    va_list ap;
    va_start(ap, req);
    void *arg = va_arg(ap, void *);
    va_end(ap);
    int r = real(fd, req, arg);
    int saved = errno;
    char p[256];
    // NVIDIA RM ioctls carry a status word the call itself reports as success; log the request
    // number so the caller's command can be matched against the open driver headers.
    out("TRACE ioctl(%d=%s, 0x%lx) = %d %s\n", fd, fdpath(fd, p, sizeof p), req, RESULT(r));
    errno = saved;
    return r;
}

int access(const char *path, int mode) {
    static int (*real)(const char *, int);
    if (!real) real = dlsym(RTLD_NEXT, "access");
    int r = real(path, mode);
    int saved = errno;
    out("TRACE access(\"%s\", %d) = %d %s\n", path ? path : "(null)", mode, RESULT(r));
    errno = saved;
    return r;
}

int stat(const char *path, struct stat *st) {
    static int (*real)(const char *, struct stat *);
    if (!real) real = dlsym(RTLD_NEXT, "stat");
    int r = real(path, st);
    int saved = errno;
    out("TRACE stat(\"%s\") = %d %s\n", path ? path : "(null)", RESULT(r));
    errno = saved;
    return r;
}

int lstat(const char *path, struct stat *st) {
    static int (*real)(const char *, struct stat *);
    if (!real) real = dlsym(RTLD_NEXT, "lstat");
    int r = real(path, st);
    int saved = errno;
    out("TRACE lstat(\"%s\") = %d %s\n", path ? path : "(null)", RESULT(r));
    errno = saved;
    return r;
}

int fstatat(int dirfd, const char *path, struct stat *st, int flags) {
    static int (*real)(int, const char *, struct stat *, int);
    if (!real) real = dlsym(RTLD_NEXT, "fstatat");
    int r = real(dirfd, path, st, flags);
    int saved = errno;
    out("TRACE fstatat(%d, \"%s\") = %d %s\n", dirfd, path ? path : "(null)", RESULT(r));
    errno = saved;
    return r;
}

ssize_t readlink(const char *path, char *buf, size_t len) {
    static ssize_t (*real)(const char *, char *, size_t);
    if (!real) real = dlsym(RTLD_NEXT, "readlink");
    ssize_t r = real(path, buf, len);
    int saved = errno;
    out("TRACE readlink(\"%s\") = %zd %s\n", path ? path : "(null)", r, r < 0 ? strerror(saved) : "ok");
    errno = saved;
    return r;
}

int socket(int domain, int type, int protocol) {
    static int (*real)(int, int, int);
    if (!real) real = dlsym(RTLD_NEXT, "socket");
    int r = real(domain, type, protocol);
    int saved = errno;
    out("TRACE socket(%d, %d, %d) = %d %s\n", domain, type, protocol, RESULT(r));
    errno = saved;
    return r;
}

int connect(int fd, const struct sockaddr *addr, socklen_t len) {
    static int (*real)(int, const struct sockaddr *, socklen_t);
    if (!real) real = dlsym(RTLD_NEXT, "connect");
    int r = real(fd, addr, len);
    int saved = errno;
    const char *where = "?";
    if (addr && addr->sa_family == AF_UNIX) where = ((const struct sockaddr_un *)addr)->sun_path;
    out("TRACE connect(%d, \"%s\") = %d %s\n", fd, where, RESULT(r));
    errno = saved;
    return r;
}

FILE *fopen(const char *path, const char *mode) {
    static FILE *(*real)(const char *, const char *);
    if (!real) real = dlsym(RTLD_NEXT, "fopen");
    FILE *r = real(path, mode);
    int saved = errno;
    out("TRACE fopen(\"%s\", \"%s\") = %s %s\n", path ? path : "(null)", mode ? mode : "", r ? "ok" : "NULL", r ? "" : strerror(saved));
    errno = saved;
    return r;
}

FILE *fopen64(const char *path, const char *mode) {
    static FILE *(*real)(const char *, const char *);
    if (!real) real = dlsym(RTLD_NEXT, "fopen64");
    FILE *r = real(path, mode);
    int saved = errno;
    out("TRACE fopen64(\"%s\", \"%s\") = %s %s\n", path ? path : "(null)", mode ? mode : "", r ? "ok" : "NULL", r ? "" : strerror(saved));
    errno = saved;
    return r;
}

#define OPEN2_LIKE(name)                                                                 \
    int name(const char *path, int flags) {                                             \
        static int (*real)(const char *, int);                                          \
        if (!real) real = dlsym(RTLD_NEXT, #name);                                      \
        int r = real(path, flags);                                                      \
        int saved = errno;                                                              \
        out("TRACE " #name "(\"%s\", 0x%x) = %d %s\n", path ? path : "(null)", flags, RESULT(r)); \
        errno = saved;                                                                  \
        return r;                                                                       \
    }
OPEN2_LIKE(__open_2)
OPEN2_LIKE(__open64_2)

char *getenv(const char *name) {
    static char *(*real)(const char *);
    if (!real) real = dlsym(RTLD_NEXT, "getenv");
    char *r = real(name);
    if (name && strcmp(name, "TRACE_FILE") != 0) out("TRACE getenv(\"%s\") = %s\n", name, r ? r : "(unset)");
    return r;
}

void *dlopen(const char *file, int mode) {
    static void *(*real)(const char *, int);
    if (!real) real = dlsym(RTLD_NEXT, "dlopen");
    void *r = real(file, mode);
    out("TRACE dlopen(\"%s\", 0x%x) = %p%s%s\n", file ? file : "(null)", mode, r, r ? "" : " ", r ? "" : "(failed)");
    return r;
}
