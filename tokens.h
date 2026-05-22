/* tokens.h -- 256-token event vocabulary for piano MIDI generation.
 *
 * The corpus is a flat byte stream where every byte is one event token:
 *
 *   [  0..87 ]  NOTE_ON  for MIDI pitch (21 + tok)     -- A0..C8, 88 keys
 *   [ 88..175]  NOTE_OFF for MIDI pitch (21 + tok-88)
 *   [176..239]  TIME_SHIFT, ((tok-176)+1) * 10 ms      -- 10ms..640ms.
 *               Longer gaps are emitted as several TIME_SHIFT tokens.
 *   [240..255]  VELOCITY,  16 bins of 8.  Emitted only when the bin
 *               changes; subsequent NOTE_ONs use the last set velocity.
 *
 * By construction every sequence of tokens is a valid piano performance.
 */
#ifndef TOKENS_H
#define TOKENS_H

#include <stdio.h>
#include <stdint.h>

#define TOK_NOTE_ON_BASE   0
#define TOK_NOTE_OFF_BASE  88
#define TOK_TIME_BASE      176
#define TOK_VEL_BASE       240

#define NUM_PITCHES        88
#define MIN_PITCH          21    /* MIDI A0 */
#define MAX_PITCH          108   /* MIDI C8 */

#define NUM_TIME_BINS      64
#define TIME_BIN_MS        10

#define NUM_VEL_BINS       16
#define VEL_BIN_SIZE       8     /* 128 / 16 */

#define VOCAB_SIZE         256


/* ---------------------------------------------------------------------- */
/*  Detokenizer: write a Standard MIDI File from an event-token array.    */
/*                                                                        */
/*  Format-0 SMF, single track.  Ticks-per-quarter = 1000 and tempo =     */
/*  1,000,000 us/quarter, so 1 tick == 1 ms and the timing math is        */
/*  trivial.  Stuck notes are NOT auto-released; the model is expected    */
/*  to emit matching NOTE_OFFs (training data does).                      */
/* ---------------------------------------------------------------------- */

static inline void midi_write_vlq_(FILE* f, uint32_t v) {
    uint8_t buf[5];
    int n = 0;
    buf[n++] = (uint8_t)(v & 0x7F);
    v >>= 7;
    while (v) {
        buf[n++] = (uint8_t)(0x80 | (v & 0x7F));
        v >>= 7;
    }
    for (int i = n - 1; i >= 0; i--) fputc(buf[i], f);
}

static inline void write_midi_from_tokens(const char* path,
                                          const unsigned char* tokens,
                                          int n)
{
    FILE* f = fopen(path, "wb");
    if (!f) return;

    /* Header: "MThd", len=6, format=0, ntrks=1, division=1000 (0x03e8). */
    static const unsigned char hdr[14] = {
        'M','T','h','d', 0,0,0,6, 0,0, 0,1, 0x03,0xe8
    };
    fwrite(hdr, 1, 14, f);

    /* Track header; we patch the length field after writing the body. */
    static const unsigned char trk[8] = { 'M','T','r','k', 0,0,0,0 };
    fwrite(trk, 1, 8, f);
    long body_start = ftell(f);

    /* Tempo meta event: 1,000,000 us/quarter. */
    fputc(0x00, f);
    fputc(0xFF, f); fputc(0x51, f); fputc(0x03, f);
    fputc(0x0F, f); fputc(0x42, f); fputc(0x40, f);

    uint32_t pending_delta = 0;
    int      vel           = 64;   /* default until a VELOCITY token appears */

    for (int i = 0; i < n; i++) {
        unsigned char t = tokens[i];
        if (t < TOK_NOTE_OFF_BASE) {                       /* NOTE_ON */
            int pitch = MIN_PITCH + (int)t;
            midi_write_vlq_(f, pending_delta); pending_delta = 0;
            fputc(0x90, f); fputc((unsigned char)pitch, f); fputc((unsigned char)vel, f);
        } else if (t < TOK_TIME_BASE) {                    /* NOTE_OFF */
            int pitch = MIN_PITCH + (int)(t - TOK_NOTE_OFF_BASE);
            midi_write_vlq_(f, pending_delta); pending_delta = 0;
            fputc(0x80, f); fputc((unsigned char)pitch, f); fputc(0x40, f);
        } else if (t < TOK_VEL_BASE) {                     /* TIME_SHIFT */
            int bin = (int)(t - TOK_TIME_BASE);            /* 0..63 */
            pending_delta += (uint32_t)((bin + 1) * TIME_BIN_MS);
        } else {                                           /* VELOCITY */
            int bin = (int)(t - TOK_VEL_BASE);             /* 0..15 */
            vel = bin * VEL_BIN_SIZE + VEL_BIN_SIZE / 2;
            if (vel > 127) vel = 127;
        }
    }

    /* End-of-track meta. */
    midi_write_vlq_(f, pending_delta);
    fputc(0xFF, f); fputc(0x2F, f); fputc(0x00, f);

    /* Patch track length. */
    long end = ftell(f);
    uint32_t body_len = (uint32_t)(end - body_start);
    fseek(f, body_start - 4, SEEK_SET);
    fputc((body_len >> 24) & 0xFF, f);
    fputc((body_len >> 16) & 0xFF, f);
    fputc((body_len >>  8) & 0xFF, f);
    fputc((body_len >>  0) & 0xFF, f);
    fclose(f);
}

#endif
