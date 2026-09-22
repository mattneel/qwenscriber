// The `QWFIX001` container that `tools/reference/gen_fixtures.py` writes and `qwenscriber-transcribe
// --dump` reads back:
//
//     offset 0   magic   8 bytes  "QWFIX001"
//     offset 8   rank    u32
//     offset 12  dims    4 x u32, unused entries written as 1
//     offset 28  payload f32 x product(dims), little endian, row major
//
// `tests/reference_check.zig` reads the same format in Zig; this reader exists because the browser
// check compares against a dump of our own runtime and needs to parse it in place.

export const FIXTURE_MAGIC = "QWFIX001";
const FIXTURE_HEADER_BYTES = 28;

export function read_fixture(bytes, label = "fixture") {
    if (bytes.byteLength < FIXTURE_HEADER_BYTES) {
        throw new Error(`${label}: ${bytes.byteLength} bytes is too short for a fixture header`);
    }
    const magic = new TextDecoder().decode(bytes.subarray(0, 8));
    if (magic !== FIXTURE_MAGIC) {
        throw new Error(`${label}: magic ${JSON.stringify(magic)} is not ${FIXTURE_MAGIC}`);
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const rank = view.getUint32(8, true);
    if (rank < 1 || rank > 4) {
        throw new Error(`${label}: rank ${rank} is outside 1..4`);
    }
    // Four entries are always written; the ones past `rank` are 1, so the product below is the
    // element count either way.
    const dims = [];
    for (let axis = 0; axis < 4; axis += 1) dims.push(view.getUint32(12 + axis * 4, true));
    const live = dims.slice(0, rank);
    const payload_words = live.reduce((product, size) => product * size, 1);
    const payload_bytes = payload_words * 4;
    if (bytes.byteLength !== FIXTURE_HEADER_BYTES + payload_bytes) {
        throw new Error(
            `${label}: ${bytes.byteLength} bytes for ${live.join("x")}, which is ` +
                `${FIXTURE_HEADER_BYTES + payload_bytes}`,
        );
    }
    return {
        rank,
        dims: live,
        values: new Float32Array(bytes.buffer, bytes.byteOffset + FIXTURE_HEADER_BYTES, payload_words),
    };
}
