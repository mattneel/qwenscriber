//! Live microphone capture: bounded at every stage, and honest about what it drops.
//!
//! The core accepts mono f32 at 16 kHz, at most `MEL_CAPACITY_SAMPLES` of it, so everything between
//! `getUserMedia` and a reader has to respect both facts. Three stages, each with an explicit bound:
//!
//!   AudioWorklet          downmix to mono, assemble fixed blocks, send only on credit
//!         |
//!         v
//!   CaptureQueue          a fixed-capacity ring of f32 samples; overflow drops the oldest
//!         |
//!         v
//!   read()                returns a copy at 16 kHz, which is what a caller feeds the runtime
//!
//! The worklet sends a block only when the main thread has granted it credit, so a stalled consumer
//! cannot make the audio thread queue without limit: it drops blocks instead and counts them. Both
//! counters are reported, because "we dropped audio" is a fact a caller has to be able to see.
//!
//! The context is requested at 16 kHz. Browsers resample the device's rate for us, once, with a
//! kernel better than anything worth writing here; a context that refuses the rate is reported, and
//! the caller can convert with `resample` instead of getting silently wrong audio.
//!
//! The worklet is registered from a blob URL built here, so the package ships no asset to locate at
//! runtime. A page with a strict `script-src` can pass `worklet_url` and serve the source itself; the
//! source is exported for exactly that.

import { MEL_CAPACITY_SAMPLES, MEL_SAMPLE_RATE_HZ, STATUS } from "../wasm/abi.ts";
import { SDK_STATUS, QwenscriberError } from "../errors.ts";

/** Frames per block sent to the main thread: about 21 ms at 16 kHz, 64 ms at 48 kHz. */
export const CAPTURE_BLOCK_FRAMES = 1024;

/** Blocks the worklet may have in flight before it must drop. Bounds the port's backlog. */
export const CAPTURE_BLOCKS_IN_FLIGHT_MAX = 4;

/** Samples the queue holds before the oldest are dropped: the core's own capacity. */
export const CAPTURE_SAMPLES_MAX = MEL_CAPACITY_SAMPLES;

/** The worklet's source, as text so the package needs no asset URL. */
export const CAPTURE_WORKLET_SOURCE = `
class QwenscriberCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const settings = (options && options.processorOptions) || {};
    this.block_frames = settings.block_frames;
    this.block = new Float32Array(this.block_frames);
    this.filled = 0;
    this.credits = settings.credits;
    this.blocks_dropped = 0;
    this.stopped = false;
    this.port.onmessage = (event) => {
      const message = event.data || {};
      if (message.credits) this.credits += message.credits;
      if (message.stop) this.stopped = true;
    };
  }

  process(inputs) {
    if (this.stopped) return false;
    const channels = inputs[0];
    if (!channels || channels.length === 0) return true;
    const frames = channels[0].length;
    const scale = channels.length > 1 ? 1 / channels.length : 1;
    for (let frame = 0; frame < frames; frame += 1) {
      let value = 0;
      for (let channel = 0; channel < channels.length; channel += 1) {
        value += channels[channel][frame];
      }
      this.block[this.filled] = value * scale;
      this.filled += 1;
      if (this.filled === this.block_frames) {
        if (this.credits > 0) {
          this.credits -= 1;
          const samples = this.block.slice();
          this.port.postMessage(
            { samples: samples, blocks_dropped: this.blocks_dropped },
            [samples.buffer],
          );
        } else {
          this.blocks_dropped += 1;
        }
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("qwenscriber-capture", QwenscriberCaptureProcessor);
`;

/** What a capture session has received and thrown away. Counts, not rates: rates are a caller's. */
export interface CaptureState {
  readonly running: boolean;
  /** Samples accepted from the worklet. */
  readonly received_samples: number;
  /** Samples the queue dropped because its reader fell behind. */
  readonly dropped_samples: number;
  /** Blocks the worklet dropped because its credits ran out. */
  readonly dropped_blocks: number;
  /**
   * Rate the captured samples are actually at, which is what `read` returns. Taken from the track the
   * browser reports, not from the context: they disagree, and the track is the one that is right.
   */
  readonly input_rate_hz: number;
  /** Rate of the context this session created, reported so the disagreement is visible. */
  readonly context_rate_hz: number;
}

export interface CaptureOptions {
  /** Rate to ask the context for. 16 kHz avoids a conversion; another rate needs `resample`. */
  readonly sample_rate_hz?: number;
  /** Samples the queue may hold. Defaults to the core's capacity; never more. */
  readonly samples_max?: number;
  /** Load the worklet from this URL instead of the source this module carries. */
  readonly worklet_url?: string;
}

/**
 * A fixed-capacity ring of f32 samples. Overflow drops the *oldest* samples: a reader that falls
 * behind wants the most recent audio, and a stream is not a recording.
 */
export class CaptureQueue {
  readonly #samples: Float32Array;
  #start = 0;
  #length = 0;
  #dropped = 0;

  constructor(capacity: number) {
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new QwenscriberError(STATUS.invalid_argument, "audio.capture.queue", {
        message: `capture queue capacity must be a positive integer, got ${capacity}`,
        context: { capacity },
      });
    }
    this.#samples = new Float32Array(capacity);
  }

  get capacity(): number {
    return this.#samples.length;
  }

  get length(): number {
    return this.#length;
  }

  get dropped_samples(): number {
    return this.#dropped;
  }

  /** Appends `samples`, dropping the oldest held samples when that is what fits. */
  push(samples: Float32Array): void {
    if (samples.length >= this.#samples.length) {
      // More than the whole queue: keep its tail, which is the part that is still news.
      const tail = samples.subarray(samples.length - this.#samples.length);
      this.#samples.set(tail);
      this.#dropped += this.#length + samples.length - this.#samples.length;
      this.#start = 0;
      this.#length = this.#samples.length;
      return;
    }
    const overflow = this.#length + samples.length - this.#samples.length;
    if (overflow > 0) {
      this.#start = (this.#start + overflow) % this.#samples.length;
      this.#length -= overflow;
      this.#dropped += overflow;
    }
    const write_at = (this.#start + this.#length) % this.#samples.length;
    const first = Math.min(samples.length, this.#samples.length - write_at);
    this.#samples.set(samples.subarray(0, first), write_at);
    if (first < samples.length) {
      this.#samples.set(samples.subarray(first), 0);
    }
    this.#length += samples.length;
  }

  /** Removes and returns up to `max_samples` as a fresh array. Short reads mean there is no more. */
  read(max_samples: number): Float32Array {
    const take = Math.max(0, Math.min(max_samples, this.#length));
    const out = new Float32Array(take);
    const first = Math.min(take, this.#samples.length - this.#start);
    out.set(this.#samples.subarray(this.#start, this.#start + first));
    if (first < take) {
      out.set(this.#samples.subarray(0, take - first), first);
    }
    this.#start = (this.#start + take) % this.#samples.length;
    this.#length -= take;
    return out;
  }
}

function workletUrlFor(options: CaptureOptions): string {
  if (options.worklet_url !== undefined) return options.worklet_url;
  const blob = new Blob([CAPTURE_WORKLET_SOURCE], { type: "application/javascript" });
  return URL.createObjectURL(blob);
}

/**
 * The rate the worklet's input actually runs at.
 *
 * A `MediaStreamAudioSourceNode` is specified to resample to its context's rate, and Chromium does
 * not: a 16 kHz track inside a 48 kHz context arrives at 16 kHz, which a caller would discover as a
 * tone that is three times too high and would rightly blame on this SDK. The track's own setting is
 * the only rate stated by the side that produced the samples, so it wins; the context rate is carried
 * in the state as well, so the disagreement is visible instead of assumed away.
 */
function inputRateFor(stream: MediaStream, context: AudioContext): number {
  const track = stream.getAudioTracks()[0];
  const reported =
    typeof track?.getSettings === "function" ? track.getSettings().sampleRate : undefined;
  if (typeof reported === "number" && Number.isFinite(reported) && reported > 0) return reported;
  return context.sampleRate;
}

function validatedSamplesMax(options: CaptureOptions): number {
  const samples_max = options.samples_max ?? CAPTURE_SAMPLES_MAX;
  if (!Number.isInteger(samples_max) || samples_max <= 0) {
    throw new QwenscriberError(STATUS.invalid_argument, "audio.capture.options", {
      message: `capture samples_max must be a positive integer, got ${samples_max}`,
      context: { samples_max },
    });
  }
  if (samples_max > CAPTURE_SAMPLES_MAX) {
    throw new QwenscriberError(STATUS.invalid_argument, "audio.capture.options", {
      message: `capture samples_max is bounded by the core's ${CAPTURE_SAMPLES_MAX} samples, got ${samples_max}`,
      context: { samples_max, samples_max_allowed: CAPTURE_SAMPLES_MAX },
    });
  }
  return samples_max;
}

/**
 * A running capture session.
 *
 * `start` acquires a microphone and owns it, down to stopping its tracks. `attach` borrows a stream
 * the caller already has, and leaving it running is the caller's business, not this class's.
 */
export class MicrophoneCapture {
  readonly #context: AudioContext;
  readonly #node: AudioWorkletNode;
  readonly #source: MediaStreamAudioSourceNode;
  readonly #stream: MediaStream;
  readonly #owns_stream: boolean;
  readonly #queue: CaptureQueue;
  readonly #input_rate_hz: number;
  readonly #context_rate_hz: number;
  #received_samples = 0;
  #dropped_blocks = 0;
  #running = true;

  private constructor(
    context: AudioContext,
    node: AudioWorkletNode,
    source: MediaStreamAudioSourceNode,
    stream: MediaStream,
    owns_stream: boolean,
    samples_max: number,
  ) {
    this.#context = context;
    this.#node = node;
    this.#source = source;
    this.#stream = stream;
    this.#owns_stream = owns_stream;
    this.#queue = new CaptureQueue(samples_max);
    this.#input_rate_hz = inputRateFor(stream, context);
    this.#context_rate_hz = context.sampleRate;
    node.port.onmessage = (event: MessageEvent) => {
      const message = event.data as { samples?: Float32Array; blocks_dropped?: number };
      if (typeof message?.blocks_dropped === "number") {
        this.#dropped_blocks = message.blocks_dropped;
      }
      const samples = message?.samples;
      if (!samples) return;
      this.#received_samples += samples.length;
      this.#queue.push(samples);
      // One block consumed, one block of credit granted: this is the whole backpressure scheme.
      node.port.postMessage({ credits: 1 });
    };
    node.port.postMessage({ credits: CAPTURE_BLOCKS_IN_FLIGHT_MAX });
  }

  /**
   * Acquires a microphone and captures from it. The stream is stopped by `stop`.
   *
   * A hostile page can ask for audio; a user decides. Nothing here retries, prompts twice, or hides
   * the rejection: `NotAllowedError` reaches the caller as it is.
   */
  static async start(options: CaptureOptions = {}): Promise<MicrophoneCapture> {
    const samples_max = validatedSamplesMax(options);
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    try {
      return await MicrophoneCapture.attachOwned(stream, true, options, samples_max);
    } catch (error) {
      for (const track of stream.getTracks()) track.stop();
      throw error;
    }
  }

  /** Captures from a stream the caller owns. `stop` detaches from it and leaves it running. */
  static async attach(
    stream: MediaStream,
    options: CaptureOptions = {},
  ): Promise<MicrophoneCapture> {
    return await MicrophoneCapture.attachOwned(stream, false, options, validatedSamplesMax(options));
  }

  private static async attachOwned(
    stream: MediaStream,
    owns_stream: boolean,
    options: CaptureOptions,
    samples_max: number,
  ): Promise<MicrophoneCapture> {
    const requested_rate = options.sample_rate_hz ?? MEL_SAMPLE_RATE_HZ;
    const context = new AudioContext({ sampleRate: requested_rate });
    try {
      const url = workletUrlFor(options);
      try {
        await context.audioWorklet.addModule(url);
      } finally {
        if (options.worklet_url === undefined) URL.revokeObjectURL(url);
      }
      const node = new AudioWorkletNode(context, "qwenscriber-capture", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        channelCount: 1,
        channelCountMode: "explicit",
        channelInterpretation: "speakers",
        processorOptions: {
          block_frames: CAPTURE_BLOCK_FRAMES,
          credits: CAPTURE_BLOCKS_IN_FLIGHT_MAX,
        },
      });
      const source = context.createMediaStreamSource(stream);
      source.connect(node);
      // A worklet with no destination is not pulled by the graph on every engine; a zero-gain sink
      // keeps it running without monitoring the microphone back into the user's speakers.
      const sink = context.createGain();
      sink.gain.value = 0;
      node.connect(sink);
      sink.connect(context.destination);
      if (context.state === "suspended") await context.resume();
      return new MicrophoneCapture(context, node, source, stream, owns_stream, samples_max);
    } catch (error) {
      await context.close();
      throw error;
    }
  }

  get state(): CaptureState {
    return {
      running: this.#running,
      received_samples: this.#received_samples,
      dropped_samples: this.#queue.dropped_samples,
      dropped_blocks: this.#dropped_blocks,
      input_rate_hz: this.#input_rate_hz,
      context_rate_hz: this.#context_rate_hz,
    };
  }

  /**
   * True when the samples are already at the core's rate, so `read` needs no conversion. Assume
   * false until checked: a caller who skips `resample` on a false positive feeds the runtime audio at
   * the wrong rate, which transcribes into confident nonsense.
   */
  get isNativeRate(): boolean {
    return this.#input_rate_hz === MEL_SAMPLE_RATE_HZ;
  }

  /**
   * Takes up to `max_samples` of mono f32 at `state.input_rate_hz`. Convert with `resample` when that
   * is not 16 kHz, which in Chromium it usually is not: the browser gives the audio device's rate.
   */
  read(max_samples: number = CAPTURE_BLOCK_FRAMES): Float32Array {
    if (!this.#running) {
      throw new QwenscriberError(SDK_STATUS.protocol, "audio.capture.read", {
        message: "capture session is stopped",
        context: { max_samples },
      });
    }
    return this.#queue.read(max_samples);
  }

  /** Detaches, closes the context, and stops the microphone if this session acquired it. */
  async stop(): Promise<void> {
    if (!this.#running) return;
    this.#running = false;
    this.#node.port.postMessage({ stop: true });
    this.#node.port.onmessage = null;
    this.#source.disconnect();
    this.#node.disconnect();
    await this.#context.close();
    if (this.#owns_stream) {
      for (const track of this.#stream.getTracks()) track.stop();
    }
  }
}
