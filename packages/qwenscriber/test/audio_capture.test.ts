//! The capture path's testable half: the bounded queue that sits between the worklet and a reader.
//!
//! What matters here is what a caller can observe — that a reader gets the samples that were pushed,
//! that falling behind costs the *oldest* audio rather than memory, and that the result of a read is
//! the reader's to keep. The worklet itself needs an audio thread and is exercised in a browser; see
//! `tests/browser/audio_capture.mjs`.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  CAPTURE_BLOCKS_IN_FLIGHT_MAX,
  CAPTURE_SAMPLES_MAX,
  CaptureQueue,
} from "../src/audio/capture.ts";
import { QwenscriberError } from "../src/errors.ts";

test("a reader gets back what was pushed, in order", () => {
  const queue = new CaptureQueue(8);
  queue.push(new Float32Array([1, 2, 3]));
  queue.push(new Float32Array([4, 5]));

  assert.equal(queue.length, 5);
  assert.deepEqual(Array.from(queue.read(3)), [1, 2, 3]);
  assert.deepEqual(Array.from(queue.read(10)), [4, 5]);
  assert.equal(queue.length, 0);
  assert.deepEqual(Array.from(queue.read(1)), []);
});

test("the ring survives wrapping past its end", () => {
  const queue = new CaptureQueue(4);
  queue.push(new Float32Array([1, 2, 3]));
  assert.deepEqual(Array.from(queue.read(2)), [1, 2]);
  // The queue's write position is now at the end, so this push wraps.
  queue.push(new Float32Array([4, 5, 6]));
  assert.equal(queue.length, 4);
  assert.deepEqual(Array.from(queue.read(4)), [3, 4, 5, 6]);
});

test("falling behind drops the oldest audio and counts what it cost", () => {
  const queue = new CaptureQueue(4);
  queue.push(new Float32Array([1, 2, 3, 4]));
  queue.push(new Float32Array([5, 6]));

  assert.equal(queue.dropped_samples, 2);
  assert.equal(queue.length, 4);
  // The most recent audio is what a live reader needs: 1 and 2 are gone, not 5 and 6.
  assert.deepEqual(Array.from(queue.read(4)), [3, 4, 5, 6]);
});

test("a push larger than the queue keeps its tail", () => {
  const queue = new CaptureQueue(3);
  queue.push(new Float32Array([1, 2]));
  queue.push(new Float32Array([3, 4, 5, 6, 7]));

  assert.deepEqual(Array.from(queue.read(3)), [5, 6, 7]);
  assert.equal(queue.dropped_samples, 4);
});

test("a read result belongs to the reader", () => {
  const queue = new CaptureQueue(4);
  queue.push(new Float32Array([1, 2, 3, 4]));

  const first = queue.read(2);
  first[0] = 99;
  const second = queue.read(2);
  assert.deepEqual(Array.from(second), [3, 4]);
});

test("a capacity that cannot hold a sample is refused", () => {
  for (const capacity of [0, -1, 1.5, Number.NaN]) {
    assert.throws(() => new CaptureQueue(capacity), QwenscriberError);
  }
  assert.equal(new CaptureQueue(1).capacity, 1);
});

test("the queue's capacity is the core's own sample budget", () => {
  // The core refuses more than 30 seconds of audio, so a queue that could hold more would only
  // accumulate audio the runtime cannot accept.
  assert.equal(CAPTURE_SAMPLES_MAX, 480000);
  assert.ok(CAPTURE_BLOCKS_IN_FLIGHT_MAX > 0);
  assert.ok(CAPTURE_BLOCKS_IN_FLIGHT_MAX * 1024 <= CAPTURE_SAMPLES_MAX);
});
