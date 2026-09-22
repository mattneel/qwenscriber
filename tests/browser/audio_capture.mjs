// Page side of the capture check: does the SDK's AudioWorklet path deliver the audio that a
// synthetic source produced, and does it stay bounded when nobody reads?
//
// A synthetic stream is enough to answer both. `createMediaStreamDestination` makes a real
// `MediaStream` out of an oscillator, so the SDK's own path runs unchanged — worklet, credits, queue,
// reader — without a microphone, a permission prompt, or a device.
//
// Two cases:
//
//   A. a 440 Hz sine comes back as a 440 Hz sine, within f32 and without dropping anything;
//   B. a reader that never reads loses the *oldest* audio, and the counters say so.
//
// Results go to `window.__result`, which the driver polls. The driver is
// `node tests/browser/audio_capture.mjs`; see its header for how Chromium is started.

const SDK = "/packages/qwenscriber/dist/index.js";

const result = { status: "pending", cases: [], error: null };
window.__result = result;

const output = document.getElementById("out");

function log(line) {
  output.textContent += `${line}\n`;
}

/** Peak, root-mean-square, upward crossings, and the spacing between them. */
function describe(samples) {
  let peak = 0;
  let sum_of_squares = 0;
  let crossings = 0;
  let last_crossing = -1;
  let gap_min = Number.POSITIVE_INFINITY;
  let gap_max = 0;
  let gap_sum = 0;
  let gap_count = 0;
  for (let index = 0; index < samples.length; index += 1) {
    const value = samples[index];
    if (!Number.isFinite(value)) return { finite: false };
    peak = Math.max(peak, Math.abs(value));
    sum_of_squares += value * value;
    if (index > 0 && samples[index - 1] <= 0 && value > 0) {
      if (last_crossing >= 0) {
        const gap = index - last_crossing;
        gap_min = Math.min(gap_min, gap);
        gap_max = Math.max(gap_max, gap);
        gap_sum += gap;
        gap_count += 1;
      }
      last_crossing = index;
      crossings += 1;
    }
  }
  return {
    finite: true,
    sample_count: samples.length,
    peak,
    rms: Math.sqrt(sum_of_squares / samples.length),
    crossings,
    // A clean tone has one gap length; a signal that repeats or drops samples does not.
    gap_min: gap_count > 0 ? gap_min : 0,
    gap_max,
    gap_mean: gap_count > 0 ? gap_sum / gap_count : 0,
  };
}

function concat(chunks) {
  let total = 0;
  for (const chunk of chunks) total += chunk.length;
  const out = new Float32Array(total);
  let at = 0;
  for (const chunk of chunks) {
    out.set(chunk, at);
    at += chunk.length;
  }
  return out;
}

/** An oscillator feeding a `MediaStream` the SDK can attach to. */
function syntheticStream(frequency_hz) {
  const context = new AudioContext({ sampleRate: 16000 });
  const oscillator = context.createOscillator();
  oscillator.type = "sine";
  oscillator.frequency.value = frequency_hz;
  const destination = context.createMediaStreamDestination();
  oscillator.connect(destination);
  oscillator.start();
  const track = destination.stream.getAudioTracks()[0];
  return {
    context,
    oscillator,
    stream: destination.stream,
    // What the source graph actually got, and what the stream says it carries: the SDK's own context
    // rate is reported beside these, so a rate mismatch is visible rather than inferred.
    context_rate_hz: context.sampleRate,
    track_rate_hz: typeof track?.getSettings === "function" ? (track.getSettings().sampleRate ?? null) : null,
  };
}

async function drain(capture, milliseconds) {
  const chunks = [];
  const deadline = performance.now() + milliseconds;
  while (performance.now() < deadline) {
    const chunk = capture.read(4096);
    if (chunk.length > 0) chunks.push(chunk);
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  return concat(chunks);
}

async function caseSine(SDKModule) {
  const { MicrophoneCapture, resample } = SDKModule;
  const source = syntheticStream(440);
  const capture = await MicrophoneCapture.attach(source.stream);
  const started = performance.now();
  const samples = await drain(capture, 800);
  const elapsed_ms = performance.now() - started;
  const state = capture.state;
  const shape = describe(samples);
  await capture.stop();
  source.oscillator.stop();
  await source.context.close();

  // The browser decides the rates: Chromium gives the context the audio device's rate while handing
  // the worklet the track's own rate, so `input_rate_hz` is what the samples are and `resample` is the
  // path this case exercises.
  const at_core_rate = resample(samples, state.input_rate_hz, 16000);
  const converted_shape = describe(at_core_rate);
  const seconds = converted_shape.sample_count / 16000;
  const measured_hz = seconds > 0 ? converted_shape.crossings / seconds : 0;
  const case_result = {
    name: "A: a synthetic sine arrives intact and converts to the core's rate",
    source_context_rate_hz: source.context_rate_hz,
    source_track_rate_hz: source.track_rate_hz,
    input_rate_hz: state.input_rate_hz,
    context_rate_hz: state.context_rate_hz,
    native_rate: capture.isNativeRate,
    received_samples: state.received_samples,
    dropped_samples: state.dropped_samples,
    dropped_blocks: state.dropped_blocks,
    // Headless Chromium paces the audio graph slower than real time; the ratio is reported because
    // it explains a sample count that looks short, and it is not asserted.
    delivered_ratio: elapsed_ms > 0 ? shape.sample_count / state.context_rate_hz / (elapsed_ms / 1000) : 0,
    shape,
    converted: { sample_count: converted_shape.sample_count, measured_hz },
    pass:
      shape.finite === true &&
      converted_shape.finite === true &&
      state.dropped_samples === 0 &&
      shape.sample_count > 4000 &&
      shape.peak > 0.3 &&
      shape.peak <= 1.0 &&
      Math.abs(shape.rms - 0.7071) < 0.15 &&
      Math.abs(measured_hz - 440) < 44,
  };
  result.cases.push(case_result);
  log(JSON.stringify(case_result));
}

async function caseBoundedReader(SDKModule) {
  const { MicrophoneCapture } = SDKModule;
  const source = syntheticStream(440);
  // A quarter second of capacity: a reader that waits a second cannot hold what arrives.
  const capture = await MicrophoneCapture.attach(source.stream, { samples_max: 4096 });
  await new Promise((resolve) => setTimeout(resolve, 900));
  const state = capture.state;
  const late = capture.read(4096);
  await capture.stop();
  source.oscillator.stop();
  await source.context.close();

  const case_result = {
    name: "B: a reader that falls behind drops the oldest audio and reports it",
    received_samples: state.received_samples,
    dropped_samples: state.dropped_samples,
    dropped_blocks: state.dropped_blocks,
    late_read_samples: late.length,
    pass:
      state.dropped_samples > 0 &&
      late.length === 4096 &&
      state.received_samples > state.dropped_samples,
  };
  result.cases.push(case_result);
  log(JSON.stringify(case_result));
}

document.getElementById("start").addEventListener("click", async () => {
  try {
    const SDKModule = await import(SDK);
    await caseSine(SDKModule);
    await caseBoundedReader(SDKModule);
    result.status = result.cases.every((entry) => entry.pass) ? "done" : "failed";
  } catch (error) {
    result.error = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    result.status = "failed";
  }
  log(`status: ${result.status}${result.error ? ` (${result.error})` : ""}`);
});
