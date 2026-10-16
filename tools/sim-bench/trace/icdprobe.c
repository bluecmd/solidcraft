// Load NVIDIA's Vulkan ICD the way the loader does, step by step, and say where it stops.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*PFN)(void);
typedef int32_t (*PFN_negotiate)(uint32_t *);
typedef PFN (*PFN_gipa)(void *, const char *);
typedef int32_t (*PFN_enum_ext)(const char *, uint32_t *, void *);
typedef int32_t (*PFN_enum_ver)(uint32_t *);

typedef struct { int32_t sType; const void *pNext; const char *app; uint32_t appv; const char *eng; uint32_t engv; uint32_t api; } AppInfo;
typedef struct { int32_t sType; const void *pNext; uint32_t flags; const AppInfo *app; uint32_t nl; const char *const *l; uint32_t ne; const char *const *e; } InstInfo;
typedef int32_t (*PFN_create)(const InstInfo *, const void *, void **);
typedef int32_t (*PFN_enum_pd)(void *, uint32_t *, void **);

int main(int argc, char **argv) {
    const char *lib = argc > 1 ? argv[1] : "libGLX_nvidia.so.0";
    dlerror();
    void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    printf("1. dlopen(%s, RTLD_NOW) = %p; dlerror: %s\n", lib, h, h ? "(none)" : dlerror());
    if (!h) return 1;
    PFN_negotiate neg = (PFN_negotiate)dlsym(h, "vk_icdNegotiateLoaderICDInterfaceVersion");
    PFN_gipa gipa = (PFN_gipa)dlsym(h, "vk_icdGetInstanceProcAddr");
    printf("2. vk_icdNegotiateLoaderICDInterfaceVersion=%p vk_icdGetInstanceProcAddr=%p vk_icdGetPhysicalDeviceProcAddr=%p\n", (void *)neg, (void *)gipa,
           dlsym(h, "vk_icdGetPhysicalDeviceProcAddr"));
    if (neg) {
        uint32_t v = 7;
        int32_t r = neg(&v);
        printf("3. negotiate(offered 7) = %d, ICD answers version %u\n", r, v);
    }
    if (!gipa) return 2;
    const char *names[] = {"vkCreateInstance", "vkEnumerateInstanceExtensionProperties", "vkEnumerateInstanceVersion", "vkEnumerateInstanceLayerProperties", 0};
    for (int i = 0; names[i]; i++) printf("4. gipa(NULL, %s) = %p\n", names[i], (void *)gipa(0, names[i]));
    PFN_enum_ver ver = (PFN_enum_ver)gipa(0, "vkEnumerateInstanceVersion");
    if (ver) {
        uint32_t v = 0;
        int32_t r = ver(&v);
        printf("5. vkEnumerateInstanceVersion = %d, %u.%u.%u\n", r, (v >> 22) & 0x7f, (v >> 12) & 0x3ff, v & 0xfff);
    }
    PFN_enum_ext ext = (PFN_enum_ext)gipa(0, "vkEnumerateInstanceExtensionProperties");
    if (ext) {
        uint32_t n = 0;
        int32_t r = ext(0, &n, 0);
        printf("6. instance extensions: r=%d count=%u\n", r, n);
    }
    PFN_create create = (PFN_create)gipa(0, "vkCreateInstance");
    if (!create) {
        printf("7. no vkCreateInstance: the ICD refused before any instance\n");
        return 3;
    }
    AppInfo app = {0, 0, "icdprobe", 1, "icdprobe", 1, (1u << 22) | (3u << 12)};
    InstInfo info = {1, 0, 0, &app, 0, 0, 0, 0};
    void *inst = 0;
    int32_t r = create(&info, 0, &inst);
    printf("7. vkCreateInstance = %d, instance %p\n", r, inst);
    if (r != 0) return 4;
    PFN_enum_pd pd = (PFN_enum_pd)gipa(inst, "vkEnumeratePhysicalDevices");
    uint32_t n = 0;
    int32_t r2 = pd ? pd(inst, &n, 0) : -99;
    printf("8. vkEnumeratePhysicalDevices = %d, %u devices\n", r2, n);
    return 0;
}
