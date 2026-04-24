// Compare two MGPT checkpoints, report per-matrix diff status.
//
// Usage:
//   check_weights <before.model> <after.model>            # verbose
//   check_weights --quiet <before.model> <after.model>    # silent, exits 0 iff every matrix changed
//
// Built by `just build-check-weights` or directly via:
//   gcc -O2 tools/check_weights.c -o bin/check_weights

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_i32(FILE* f) { int v; if (fread(&v, 4, 1, f) != 1) return -1; return v; }
static unsigned read_u32(FILE* f) { unsigned v; if (fread(&v, 4, 1, f) != 1) return 0; return v; }

typedef struct { char name[32]; long off; long size; } Mat;

int main(int argc, char** argv) {
    int quiet = 0, arg_i = 1;
    if (argc >= 2 && strcmp(argv[1], "--quiet") == 0) { quiet = 1; arg_i = 2; }
    if (argc < arg_i + 2) {
        fprintf(stderr, "Usage: %s [--quiet] <before.model> <after.model>\n", argv[0]);
        return 2;
    }
    const char* path1 = argv[arg_i];
    const char* path2 = argv[arg_i + 1];
    FILE* f1 = fopen(path1, "rb");
    FILE* f2 = fopen(path2, "rb");
    if (!f1 || !f2) { perror("open"); return 2; }

    unsigned magic = read_u32(f1);
    int D = read_i32(f1);
    int H = read_i32(f1);
    int L = read_i32(f1);
    int F = read_i32(f1);
    int V = read_i32(f1);
    int seq = read_i32(f1);
    (void)magic; (void)H; (void)F; (void)V; (void)seq;

    Mat mats[256]; int nm = 0;
    fseek(f1, 28, SEEK_SET);

    #define RECORD(name_) do { \
        long off = ftell(f1); \
        int rows = read_i32(f1); \
        int cols = read_i32(f1); \
        long bytes = (long)rows * cols * 4; \
        snprintf(mats[nm].name, sizeof(mats[nm].name), "%s", name_); \
        mats[nm].off = off; mats[nm].size = bytes + 8; nm++; \
        fseek(f1, bytes, SEEK_CUR); \
    } while(0)

    RECORD("token_emb");
    for (int i = 0; i < L; i++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "L%d.wq_w", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.wq_b", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.wk_w", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.wk_b", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.wv_w", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.wv_b", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.wo_w", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.wo_b", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.ln1_g", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.ln1_b", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.l1_w", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.l1_b", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.l2_w", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.l2_b", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.ln2_g", i); RECORD(buf);
        snprintf(buf, sizeof(buf), "L%d.ln2_b", i); RECORD(buf);
    }
    RECORD("final_gamma"); RECORD("final_beta");
    RECORD("out_w"); RECORD("out_b");
    fclose(f1);

    f1 = fopen(path1, "rb");
    f2 = fopen(path2, "rb");
    if (!quiet) {
        printf("%-16s %10s %14s %14s\n", "matrix", "size_bytes", "changed_bytes", "max_abs_diff");
    }
    int n_frozen = 0;
    char first_frozen[32] = {0};
    for (int i = 0; i < nm; i++) {
        long size = mats[i].size;
        fseek(f1, mats[i].off, SEEK_SET);
        fseek(f2, mats[i].off, SEEK_SET);
        unsigned char* b1 = (unsigned char*)malloc(size);
        unsigned char* b2 = (unsigned char*)malloc(size);
        size_t r1 = fread(b1, 1, size, f1);
        size_t r2 = fread(b2, 1, size, f2);
        (void)r1; (void)r2;
        long changed = 0;
        float max_diff = 0.0f;
        for (long j = 8; j < size; j += 4) {
            float v1, v2;
            memcpy(&v1, b1+j, 4);
            memcpy(&v2, b2+j, 4);
            float d = v1 - v2; if (d < 0) d = -d;
            if (d > 0) changed++;
            if (d > max_diff) max_diff = d;
        }
        if (!quiet) {
            printf("%-16s %10ld %14ld %14.6e\n", mats[i].name, size-8, changed*4, max_diff);
        }
        if (changed == 0) {
            if (n_frozen == 0) snprintf(first_frozen, sizeof(first_frozen), "%s", mats[i].name);
            n_frozen++;
        }
        free(b1); free(b2);
    }
    fclose(f1); fclose(f2);

    if (n_frozen > 0) {
        if (quiet) {
            fprintf(stderr, "FAIL: %d weight matrices are frozen (first: %s)\n", n_frozen, first_frozen);
        } else {
            printf("\nFAIL: %d weight matrices are frozen (first: %s)\n", n_frozen, first_frozen);
        }
        return 1;
    }
    if (!quiet) {
        printf("\nOK: all %d weight matrices changed\n", nm);
    }
    return 0;
}
