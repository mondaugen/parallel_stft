#include <pthreads.h>
#include "ipp.h"

/* All in one struct. */
typedef struct result_list_t result_list_t;
struct {
    Ipp32f *x;
    size_t len;
    int thread_num;
    result_list_t *next;
} result_list_t;

pthread_mutex_t rl_mutex;

#define N_THREADS 128
#define DFT_ORDER 12
#define DFT_LEN (1<<DFT_ORDER)
#define HOP_SIZE (DFT_LEN/4)

result_list_t *results = NULL;
pthread_t threads[N_THREADS];
int thread_state[N_THREADS];
IppsFFTSpec_R_32f *rfft_specs[N_THREADS];
Ipp8u *fft_bufs[N_THREADS];

int get_free_thread()
{
    int n = 0;
    while (n < N_THREADS) {
        if (thread_state[n]) {
            return n;
        }
    }
    return -1;
}

void obtain_thread(int n)
{
    if (n >= 0) { thread_state[n] = 0; }
}

void release_thread(int n)
{
    if (n >= 0) { thread_state[n] = 1; }
}

void *
thread_fun(void *d)
{
    result_list_t *rl = (result_list_t*)d;
    ippsFFTFwd_RToPerm_32f_I(d->x,rfft_specs[d->thread_num],fft_bufs[d->thread_num]);
    pthread_mutex_lock(&rl_mutex); /* Wait until free */
    rl->next = results;
    results = rl;
    release_thread(d->thread_num);
    pthread_mutex_unlock(&rl_mutex);
    return d;
}

int main (void)
{
    Ipp32f *x = malloc(sizeof(Ipp32f)*SIGNAL_LEN);
    size_t n;
    for (n = 0; n < SIGNAL_LEN; n++ ) { x[n] = random()/(Ipp32f)RAND_MAX*2.-1.; }
    int spec_size, int spec_buf_size, int buf_size;
    ippsFFTGetSize_R_32f(DFT_ORDER,IPP_FFT_NODIV_BY_ANY,0,&spec_size,&spec_buf_size,&buf_size);
    for (n = 0; n < N_THREADS; n++) {
        // Allocate buffers and initialize FFT structures...
    }
    for (n = 0; n < (SIGNAL_LEN/DFT_LEN)*DFT_LEN; n += HOP_SIZE) {
        pthread_mutex_lock(&rl_mutex);
        int m = get_free_thread();
        if (m >= 0) {
            obtain_thread(m);
            result_list_t *r = malloc(sizeof(result_list_t)+sizeof(Ipp32f)*DFT_LEN);
            if (r) {
            *r = (result_list_t) {
                .len = DFT_LEN,
                .thread_num = m,
            };
            Ipp32f *tmp = (Ipp32f*)(r+1);
            pthread_create(&threads[m],NULL,thread_fun,(void*)r);
            } else { release_thread(m); }
        }
        pthread_mutex_unlock(&rl_mutex);
    }
    for (n = 0; n < N_THREADS; n++) {
        pthread_join(&threads[n],NULL);
    }
    n = 0;
    while (results) {
        /* Out of order, for now */
        char buf[1024];
        sprintf(buf,"/tmp/stft_frame_perm-%zu.f32",n);
        FILE *f = fopen(buf,"w");
        if (!f) { continue; }
        fwrite(results->x,sizeof(Ipp32f),results->len,f);
        fclose(f);
        results = results->next;
        n++;
    }
    return 0;
}




