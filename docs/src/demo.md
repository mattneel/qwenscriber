# Live demo

This page runs Qwenscriber in your browser. It is the repository's own example page
(`examples/browser`), staged into the book by `tools/book/stage-demo.mjs`, so there is one demo driver
rather than a documentation copy that drifts from the code.

What it does, in order: probe the host's capabilities, load the `wasm32-freestanding` core in a
Worker and run its self-test, decode and resample audio, compute log-mel features, detokenize ids,
time the preprocessing path, and — when the browser has a WebGPU adapter — acquire one, plan weight
uploads against its real limits, and run `matmul_q4` against the same CPU reference the GPU harness
uses.

What it does **not** do: produce a transcript — not because the runtime cannot, but because a
transcript needs a converted model directory of several hundred megabytes and the book ships no model
artifacts. The decode path is exercised by `packages/qwenscriber/test/decode_model.test.ts` against a
converted checkpoint on a machine that has one; the last section of the page shows the typed error a
caller gets without one, rather than a plausible-looking fake.

The page also reports the adapter it actually got and the upload plan for a deliberately impossible
manifest, because "how many buffers does a quantized 1.7B model need" is the question that decides
whether a browser can load it at all.

<iframe
  src="demo/index.html"
  title="Qwenscriber live demo"
  style="width: 100%; height: 1200px; border: 1px solid #3a3f47; border-radius: 6px"
  loading="lazy"
></iframe>

If the frame is too small, open [the demo in its own tab](demo/index.html).

## Running it locally

```sh
cd packages/qwenscriber && npm install && npm run build
node tools/book/stage-demo.mjs
mdbook serve docs --open
```

The page needs an http origin: `file://` is not a secure context, so workers and the `fetch` of the
module both fail there.

## Reading the numbers

`preprocess` reports wall time per clip and its realtime factor, which is the ratio of wall time to
audio duration — below 1.0 means faster than real time. The self-test's `quant_hash` and `mel_hash`
are the same constants the native `qwenscriber-selftest` prints, so a page that shows them is running
the same arithmetic as the host build, not an approximation of it.

The WebGPU section reports the adapter it actually got (`high-performance` is requested first), the
adapter's measured limits, and the upload plan for a tensor and for a deliberately impossible
manifest, because "how many buffers does this need" is the question that decides whether a quantized
1.7B model can load at all.
